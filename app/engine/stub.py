from __future__ import annotations

import base64
import binascii
import hashlib
import re
import struct
import time
import zlib

from app.config import ModelSettings
from app.engine.common import GeneratedImagePayload, ImageJob, ImageResult


_PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
_SIZE_RE = re.compile(r"^(\d{2,4})x(\d{2,4})$")


class StubImageRuntime:
    def __init__(self, model_name: str, settings: ModelSettings) -> None:
        self.model_name = model_name
        self.settings = settings

    async def complete(self, job: ImageJob) -> ImageResult:
        started_at = time.perf_counter()
        width, height = _parse_size(job.size)
        if job.operation == "edit":
            _validate_edit_inputs(job)
        images = []
        for index in range(job.n):
            key = f"{job.model}:{job.operation}:{job.prompt}:{job.seed}:{index}:{len(job.images)}"
            png_bytes = _png_bytes(width, height, key)
            images.append(
                GeneratedImagePayload(
                    b64_json=base64.b64encode(png_bytes).decode("ascii"),
                    revised_prompt=f"{job.prompt} [stub image {index + 1}]",
                )
            )
        return ImageResult(
            images=images,
            metrics={
                "backend": "stub",
                "backend_inference_wall_ms": (time.perf_counter() - started_at) * 1000,
                "operation": job.operation,
                "image_count": len(images),
                "input_image_count": len(job.images),
                "width": width,
                "height": height,
            },
        )


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


def _validate_edit_inputs(job: ImageJob) -> None:
    if not job.images:
        raise ValueError("image edit requests require at least one input image")
    for image in job.images:
        if not image.data_url.startswith("data:image/"):
            raise ValueError("input images must be data URLs with an image media type")
        _header, separator, payload = image.data_url.partition(",")
        if not separator:
            raise ValueError("input image data URL is missing base64 payload")
        try:
            base64.b64decode(payload, validate=True)
        except binascii.Error as exc:
            raise ValueError("input image data URL payload is not valid base64") from exc


def _chunk(chunk_type: bytes, payload: bytes) -> bytes:
    crc = binascii.crc32(chunk_type + payload) & 0xFFFFFFFF
    return struct.pack(">I", len(payload)) + chunk_type + payload + struct.pack(">I", crc)


def _png_bytes(width: int, height: int, key: str) -> bytes:
    digest = hashlib.sha256(key.encode("utf-8")).digest()
    rows = bytearray()
    for y in range(height):
        rows.append(0)
        for x in range(width):
            rows.append((digest[0] + x * 3 + y * 5) % 256)
            rows.append((digest[7] + x * 2 + y * 11) % 256)
            rows.append((digest[15] + x * 13 + y) % 256)
    ihdr = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    idat = zlib.compress(bytes(rows), level=6)
    return _PNG_SIGNATURE + _chunk(b"IHDR", ihdr) + _chunk(b"IDAT", idat) + _chunk(b"IEND", b"")
