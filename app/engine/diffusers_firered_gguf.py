from __future__ import annotations

import asyncio
import base64
import binascii
import io
import re
import time
from pathlib import Path
from typing import Any

from PIL import Image, UnidentifiedImageError

from app.config import ModelSettings
from app.engine.common import GeneratedImagePayload, ImageJob, ImageResult


BASE_MODEL_ID = "FireRedTeam/FireRed-Image-Edit-1.1"
_SIZE_RE = re.compile(r"^(\d{2,4})x(\d{2,4})$")


class DiffusersFireRedGgufRuntime:
    def __init__(self, model_name: str, settings: ModelSettings) -> None:
        if not settings.model_path:
            raise ValueError(f"model_path is required for {settings.backend}")

        started_at = time.perf_counter()
        import torch
        from diffusers import GGUFQuantizationConfig, QwenImageEditPipeline, QwenImageTransformer2DModel

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required for diffusers_firered_gguf")

        self.model_name = model_name
        self.settings = settings
        self._torch = torch
        self._device = "cuda"
        self._dtype = torch.bfloat16
        base_model_path = settings.base_model_path or BASE_MODEL_ID
        transformer = QwenImageTransformer2DModel.from_single_file(
            _resolve_model_path(settings.model_path),
            quantization_config=GGUFQuantizationConfig(compute_dtype=self._dtype),
            torch_dtype=self._dtype,
            config=base_model_path,
            subfolder="transformer",
        )
        self._pipe = QwenImageEditPipeline.from_pretrained(
            base_model_path,
            transformer=transformer,
            torch_dtype=self._dtype,
            local_files_only=True,
        )
        self._pipe.vae.enable_tiling()
        self._pipe.vae.enable_slicing()
        self._pipe.to(self._device)
        self._load_wall_ms = (time.perf_counter() - started_at) * 1000

    async def complete(self, job: ImageJob) -> ImageResult:
        return await asyncio.to_thread(self._complete_sync, job)

    def _complete_sync(self, job: ImageJob) -> ImageResult:
        if job.operation != "edit":
            raise ValueError(f"{self.model_name} supports image edits only")

        started_at = time.perf_counter()
        width, height = _parse_size(job.size)
        steps = _metadata_int(job.metadata, "steps", default=40, minimum=1, maximum=80)
        true_cfg_scale = _metadata_float(job.metadata, "guidance", default=4.0, minimum=0.0, maximum=20.0)
        input_images = _decode_images(job)

        images: list[GeneratedImagePayload] = []
        for index in range(job.n):
            generator = None
            if job.seed is not None:
                generator = self._torch.Generator(device=self._device).manual_seed(job.seed + index)

            output = self._pipe(
                image=input_images[0] if len(input_images) == 1 else input_images,
                prompt=job.prompt,
                negative_prompt=" ",
                height=height,
                width=width,
                num_inference_steps=steps,
                true_cfg_scale=true_cfg_scale,
                generator=generator,
            ).images[0]
            images.append(
                GeneratedImagePayload(
                    b64_json=_encode_png(output),
                    revised_prompt=job.prompt,
                )
            )

        self._torch.cuda.synchronize()
        return ImageResult(
            images=images,
            metrics={
                "backend": self.settings.backend,
                "backend_load_wall_ms": self._load_wall_ms,
                "backend_inference_wall_ms": (time.perf_counter() - started_at) * 1000,
                "operation": job.operation,
                "image_count": len(images),
                "input_image_count": len(job.images),
                "width": width,
                "height": height,
                "steps": steps,
                "guidance": true_cfg_scale,
                "device": self._device,
                "torch_dtype": "bfloat16",
            },
        )


def _resolve_model_path(model_path: str) -> str:
    path = Path(model_path)
    if path.exists():
        return str(path)
    if "::" not in model_path:
        return model_path

    from huggingface_hub import hf_hub_download

    repo_id, filename = model_path.split("::", maxsplit=1)
    return hf_hub_download(repo_id=repo_id, filename=filename)


def _parse_size(size: str) -> tuple[int, int]:
    if size == "auto":
        return (512, 512)
    match = _SIZE_RE.match(size)
    if match is None:
        raise ValueError("size must be 'auto' or '<width>x<height>'")
    width = int(match.group(1))
    height = int(match.group(2))
    if width < 64 or height < 64 or width > 1024 or height > 1024:
        raise ValueError("size must be between 64x64 and 1024x1024")
    return (width, height)


def _metadata_int(metadata: dict[str, Any], key: str, *, default: int, minimum: int, maximum: int) -> int:
    try:
        parsed = int(metadata.get(key, default))
    except (TypeError, ValueError):
        return default
    return min(maximum, max(minimum, parsed))


def _metadata_float(metadata: dict[str, Any], key: str, *, default: float, minimum: float, maximum: float) -> float:
    try:
        parsed = float(metadata.get(key, default))
    except (TypeError, ValueError):
        return default
    return min(maximum, max(minimum, parsed))


def _decode_images(job: ImageJob) -> list[Image.Image]:
    if not job.images:
        raise ValueError("FireRed edit requests require at least one input image")

    decoded = []
    for image in job.images:
        if not image.data_url.startswith("data:image/"):
            raise ValueError("input images must be data URLs with an image media type")
        _header, separator, payload = image.data_url.partition(",")
        if not separator:
            raise ValueError("input image data URL is missing base64 payload")
        try:
            image_bytes = base64.b64decode(payload, validate=True)
        except binascii.Error as exc:
            raise ValueError("input image data URL payload is not valid base64") from exc
        try:
            decoded.append(Image.open(io.BytesIO(image_bytes)).convert("RGB"))
        except UnidentifiedImageError as exc:
            raise ValueError("input image data URL payload is not a valid image") from exc
    return decoded


def _encode_png(image: Image.Image) -> str:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("ascii")
