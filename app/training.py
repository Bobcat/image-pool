from __future__ import annotations

import copy
import importlib.util
import json
import random
import threading
import traceback
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fastapi import HTTPException

from app.engine.common import release_torch_cuda_memory
from app.engine.router import ImageRouterEngine
from app.schemas import FluxLoraTrainingStartRequest

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp"}
TRAINING_SIZE_MULTIPLE = 16
TRAINING_CHECKPOINT_INTERVAL = 500
FLUX_TRAINING_PATCH_SIZE = 2
LORA_TARGET_MODULES = [
    "to_k",
    "to_q",
    "to_v",
    "to_out.0",
    "to_qkv_mlp_proj",
    *[f"single_transformer_blocks.{index}.attn.to_out" for index in range(24)],
]
TRAINING_GUIDANCE_SCALE = 3.5


@dataclass(frozen=True)
class TrainingExample:
    image_path: Path
    caption: str


@dataclass(frozen=True)
class TrainingBucket:
    width: int
    height: int

    @property
    def label(self) -> str:
        return f"{self.width}x{self.height}"


def _initial_state() -> dict[str, Any]:
    return {
        "status": "idle",
        "run_id": "",
        "pid": None,
        "returncode": None,
        "started_at": "",
        "completed_at": "",
        "output_path": "",
        "log_tail": "",
        "message": "",
        "progress": {
            "step": 0,
            "steps": 0,
            "loss": None,
            "learning_rate": None,
        },
    }


_LOCK = threading.Lock()
_STATE = _initial_state()
_THREAD: threading.Thread | None = None
_STOP_EVENT: threading.Event | None = None


def training_status() -> dict[str, Any]:
    with _LOCK:
        run = _state_snapshot_locked()
    return {
        "backend": _backend_status(),
        "run": run,
    }


def start_flux_lora_training(
    engine: ImageRouterEngine,
    request: FluxLoraTrainingStartRequest,
) -> dict[str, Any]:
    global _STATE, _STOP_EVENT, _THREAD

    model_payload = _model_status(engine, request.model)
    dataset_payload = _dataset_status(Path(request.dataset_path))
    output_path = Path(request.output_path).expanduser()
    backend_payload = _backend_status()
    preflight = {
        "backend": backend_payload,
        "model": model_payload,
        "dataset": dataset_payload,
        "output_path": str(output_path),
    }

    if not dataset_payload["ready"]:
        raise HTTPException(
            status_code=409,
            detail={
                "error": "training_dataset_not_ready",
                "message": "Dataset must contain image files with matching .txt captions.",
                "preflight": preflight,
            },
        )
    if not model_payload["ready"]:
        raise HTTPException(
            status_code=409,
            detail={
                "error": "training_model_not_ready",
                "message": model_payload["message"],
                "preflight": preflight,
            },
        )
    if not backend_payload["available"]:
        raise HTTPException(
            status_code=501,
            detail={
                "error": "training_backend_unavailable",
                "message": backend_payload["message"],
                "preflight": preflight,
            },
        )

    with _LOCK:
        if _STATE["status"] in {"running", "stopping"}:
            raise HTTPException(
                status_code=409,
                detail={
                    "error": "training_run_active",
                    "message": "A Flux LoRA training run is already active.",
                    "run": _state_snapshot_locked(),
                },
            )

        run_id = _run_id()
        run_dir = output_path / run_id
        stop_event = threading.Event()
        _STATE = _initial_state()
        _STATE.update(
            {
                "status": "running",
                "run_id": run_id,
                "started_at": _utc_now(),
                "output_path": str(run_dir),
                "message": "Starting Flux LoRA training.",
                "progress": {
                    "step": 0,
                    "steps": request.steps,
                    "loss": None,
                    "learning_rate": request.learning_rate,
                },
            }
        )
        _STOP_EVENT = stop_event

    thread = _start_training_thread(request, model_payload, run_dir, stop_event)
    with _LOCK:
        _THREAD = thread

    return training_status()


def stop_training() -> dict[str, Any]:
    with _LOCK:
        stop_event = _STOP_EVENT
        is_running = _STATE["status"] == "running"
        if is_running:
            _STATE["status"] = "stopping"
            _STATE["message"] = "Stop requested; finishing the current step."
    if is_running and stop_event is not None:
        stop_event.set()
    return training_status()


def _backend_status() -> dict[str, Any]:
    missing_dependencies = [
        module_name
        for module_name in ("diffusers", "peft", "torch", "PIL")
        if importlib.util.find_spec(module_name) is None
    ]
    implemented = True
    if not implemented:
        message = "Flux LoRA trainer is not implemented in image-pool yet."
    elif missing_dependencies:
        message = f"Missing training dependency: {', '.join(missing_dependencies)}."
    else:
        message = "Flux LoRA trainer is available."
    return {
        "id": "diffusers_flux2_lora",
        "implemented": implemented,
        "available": implemented and not missing_dependencies,
        "missing_dependencies": missing_dependencies,
        "message": message,
    }


def _model_status(engine: ImageRouterEngine, model_name: str) -> dict[str, Any]:
    try:
        settings = engine.settings.engine.models[model_name]
    except KeyError:
        return {
            "ready": False,
            "model": model_name,
            "backend": "",
            "model_path": "",
            "message": f"Unknown model: {model_name}",
        }

    model_path = Path(settings.model_path or "").expanduser()
    if settings.backend != "diffusers_flux2_klein":
        message = f"Model backend must be diffusers_flux2_klein, got {settings.backend}."
        ready = False
    elif not settings.model_path:
        message = "Model has no model_path configured."
        ready = False
    elif not model_path.exists():
        message = f"Model path does not exist: {model_path}"
        ready = False
    else:
        message = "Model is ready for training preflight."
        ready = True

    return {
        "ready": ready,
        "model": model_name,
        "backend": settings.backend,
        "model_path": str(model_path) if settings.model_path else "",
        "message": message,
    }


def _dataset_status(dataset_path: Path) -> dict[str, Any]:
    path = dataset_path.expanduser()
    if not path.exists() or not path.is_dir():
        return {
            "ready": False,
            "path": str(path),
            "image_count": 0,
            "caption_count": 0,
            "missing_captions": [],
            "message": f"Dataset path does not exist: {path}",
        }

    image_paths = sorted(item for item in path.iterdir() if item.suffix.lower() in IMAGE_EXTENSIONS)
    caption_paths = [image_path.with_suffix(".txt") for image_path in image_paths]
    missing_captions = [caption_path.name for caption_path in caption_paths if not _has_caption(caption_path)]
    ready = bool(image_paths) and not missing_captions
    return {
        "ready": ready,
        "path": str(path),
        "image_count": len(image_paths),
        "caption_count": len(caption_paths) - len(missing_captions),
        "missing_captions": missing_captions,
        "message": "Dataset is ready." if ready else "Dataset is missing image captions.",
    }


def _has_caption(path: Path) -> bool:
    return path.exists() and path.read_text(encoding="utf-8").strip() != ""


def _start_training_thread(
    request: FluxLoraTrainingStartRequest,
    model_payload: dict[str, Any],
    run_dir: Path,
    stop_event: threading.Event,
) -> threading.Thread:
    thread = threading.Thread(
        target=_run_training_worker,
        args=(request, Path(model_payload["model_path"]), run_dir, stop_event),
        name=f"flux-lora-training-{run_dir.name}",
        daemon=True,
    )
    thread.start()
    return thread


def _run_training_worker(
    request: FluxLoraTrainingStartRequest,
    model_path: Path,
    run_dir: Path,
    stop_event: threading.Event,
) -> None:
    global _THREAD, _STOP_EVENT

    log_path = run_dir / "train.log"
    try:
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "request.json").write_text(
            json.dumps(request.model_dump(), indent=2, sort_keys=True),
            encoding="utf-8",
        )
        stopped = _run_flux_lora_training(request, model_path, run_dir, log_path, stop_event)
    except Exception as exc:  # pragma: no cover - exercised by real trainer failures.
        _append_log(log_path, f"ERROR: {exc}\n{traceback.format_exc()}")
        _update_state(
            status="failed",
            completed_at=_utc_now(),
            returncode=1,
            message=str(exc),
            log_tail=_read_log_tail(log_path),
        )
    else:
        _update_state(
            status="stopped" if stopped else "completed",
            completed_at=_utc_now(),
            returncode=0,
            message="Training stopped." if stopped else "Training completed.",
            log_tail=_read_log_tail(log_path),
        )
    finally:
        with _LOCK:
            _THREAD = None
            _STOP_EVENT = None


def _run_flux_lora_training(
    request: FluxLoraTrainingStartRequest,
    model_path: Path,
    run_dir: Path,
    log_path: Path,
    stop_event: threading.Event,
) -> bool:
    import torch
    from diffusers import Flux2KleinPipeline
    from peft import LoraConfig
    from peft.utils import get_peft_model_state_dict
    from PIL import Image
    from torch.nn import functional as F

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for Flux LoRA training.")

    examples = _dataset_examples(Path(request.dataset_path))
    resolution_buckets = _training_resolutions(request.resolution)
    device = torch.device("cuda")
    weight_dtype = torch.bfloat16
    generator = torch.Generator(device=device)
    randomizer = random.Random()
    pipe = None

    def save_lora_weights(save_dir: Path) -> None:
        save_dir.mkdir(parents=True, exist_ok=True)
        transformer_lora_layers = get_peft_model_state_dict(pipe.transformer)
        transformer_lora_layers = {
            key: value.detach().cpu().contiguous() if isinstance(value, torch.Tensor) else value
            for key, value in transformer_lora_layers.items()
        }
        Flux2KleinPipeline.save_lora_weights(
            save_directory=str(save_dir),
            transformer_lora_layers=transformer_lora_layers,
            safe_serialization=True,
        )

    _append_log(log_path, f"Loading Flux2KleinPipeline from {model_path}")
    _append_log(
        log_path,
        (
            "Training config: "
            f"images={len(examples)}, steps={request.steps}, resolution_buckets={resolution_buckets}, "
            f"batch_size={request.batch_size}, rank={request.rank}, alpha={request.alpha}, "
            f"learning_rate={request.learning_rate}, checkpoint_interval={TRAINING_CHECKPOINT_INTERVAL}, "
            "timestep_type=shift"
        ),
    )

    try:
        pipe = Flux2KleinPipeline.from_pretrained(
            str(model_path),
            torch_dtype=weight_dtype,
            local_files_only=True,
        )
        pipe.to(device)
        pipe.vae.requires_grad_(False)
        pipe.text_encoder.requires_grad_(False)
        pipe.transformer.requires_grad_(False)
        pipe.vae.eval()
        pipe.text_encoder.eval()
        pipe.transformer.train()
        if hasattr(pipe.transformer, "enable_gradient_checkpointing"):
            pipe.transformer.enable_gradient_checkpointing()

        lora_config = LoraConfig(
            r=request.rank,
            lora_alpha=request.alpha,
            init_lora_weights="gaussian",
            target_modules=LORA_TARGET_MODULES,
        )
        pipe.transformer.add_adapter(lora_config)
        trainable_params = [parameter for parameter in pipe.transformer.parameters() if parameter.requires_grad]
        if not trainable_params:
            raise RuntimeError("LoRA adapter did not expose trainable parameters.")

        optimizer = torch.optim.AdamW(trainable_params, lr=request.learning_rate)
        noise_scheduler = copy.deepcopy(pipe.scheduler)
        train_timestep_count = int(getattr(noise_scheduler.config, "num_train_timesteps", 1000))
        _append_log(log_path, f"Using {train_timestep_count} scheduler training timesteps.")

        for step in range(1, request.steps + 1):
            if stop_event.is_set():
                _append_log(log_path, f"Stop requested before step {step}.")
                return True

            batch = [randomizer.choice(examples) for _ in range(request.batch_size)]
            bucket = _select_training_bucket(Image, batch[0].image_path, resolution_buckets, randomizer)
            captions = [item.caption for item in batch]

            with torch.no_grad():
                prompt_embeds, text_ids = pipe.encode_prompt(
                    prompt=captions,
                    device=device,
                    max_sequence_length=512,
                    text_encoder_out_layers=(9, 18, 27),
                )
                prompt_embeds = prompt_embeds.to(device=device, dtype=pipe.transformer.dtype)
                text_ids = text_ids.to(device=device)
                pixel_values = torch.cat(
                    [_load_training_image(Image, pipe, item.image_path, bucket) for item in batch],
                    dim=0,
                ).to(device=device, dtype=pipe.vae.dtype)
                model_input = pipe._encode_vae_image(pixel_values, generator=generator)

            shift_mu = _set_shifted_training_timesteps(noise_scheduler, model_input, train_timestep_count, device)
            model_input_ids = pipe._prepare_latent_ids(model_input).to(device=device)
            noise = torch.randn_like(model_input)
            timesteps, timestep_indices = _sample_timesteps(torch, noise_scheduler, model_input.shape[0], device)
            sigmas = _sigmas_for_timesteps(torch, noise_scheduler, timestep_indices, model_input.ndim, model_input.dtype, device)
            noisy_model_input = (1.0 - sigmas) * model_input + sigmas * noise
            packed_noisy_model_input = pipe._pack_latents(noisy_model_input).to(dtype=pipe.transformer.dtype)

            guidance = None
            if getattr(pipe.transformer.config, "guidance_embeds", False):
                guidance = torch.full(
                    (model_input.shape[0],),
                    TRAINING_GUIDANCE_SCALE,
                    device=device,
                    dtype=model_input.dtype,
                )

            model_pred = pipe.transformer(
                hidden_states=packed_noisy_model_input,
                timestep=timesteps / 1000,
                guidance=guidance,
                encoder_hidden_states=prompt_embeds,
                txt_ids=text_ids,
                img_ids=model_input_ids,
                return_dict=False,
            )[0]
            model_pred = model_pred[:, : packed_noisy_model_input.size(1) :]
            model_pred = pipe._unpack_latents_with_ids(
                model_pred,
                model_input_ids,
                model_input.shape[-2],
                model_input.shape[-1],
            )

            target = noise - model_input
            loss = F.mse_loss(model_pred.float(), target.float())
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
            optimizer.step()

            if step % TRAINING_CHECKPOINT_INTERVAL == 0:
                checkpoint_dir = _checkpoint_dir(run_dir, step)
                save_lora_weights(checkpoint_dir)
                _append_log(log_path, f"Saved checkpoint LoRA weights to {checkpoint_dir}")

            if step == 1 or step % 10 == 0 or step == request.steps:
                loss_value = float(loss.detach().item())
                shift_label = f" shift_mu={shift_mu:.4f}" if shift_mu is not None else ""
                _append_log(log_path, f"step={step}/{request.steps} bucket={bucket.label}{shift_label} loss={loss_value:.6f}")
                _update_state(
                    progress={
                        "step": step,
                        "steps": request.steps,
                        "loss": loss_value,
                        "learning_rate": request.learning_rate,
                    },
                    message=f"Training step {step}/{request.steps}.",
                    log_tail=_read_log_tail(log_path),
                )

        save_lora_weights(run_dir)
        _append_log(log_path, f"Saved LoRA weights to {run_dir}")
        return False
    finally:
        if pipe is not None:
            pipe.to("cpu")
        del pipe
        release_torch_cuda_memory(torch)


def _dataset_examples(dataset_path: Path) -> list[TrainingExample]:
    examples: list[TrainingExample] = []
    for image_path in sorted(item for item in dataset_path.expanduser().iterdir() if item.suffix.lower() in IMAGE_EXTENSIONS):
        caption_path = image_path.with_suffix(".txt")
        caption = caption_path.read_text(encoding="utf-8").strip()
        if caption:
            examples.append(TrainingExample(image_path=image_path, caption=caption))
    if not examples:
        raise RuntimeError(f"No captioned training images found in {dataset_path}.")
    return examples


def _load_training_image(image_module, pipe, image_path: Path, bucket: TrainingBucket):
    with image_module.open(image_path) as image:
        image = image.convert("RGB")
        return pipe.image_processor.preprocess(
            image,
            height=bucket.height,
            width=bucket.width,
            resize_mode="crop",
        )


def _training_resolutions(resolutions: list[int]) -> list[int]:
    valid = sorted({_round_to_multiple(value, TRAINING_SIZE_MULTIPLE) for value in resolutions if value >= 256})
    return valid or [512]


def _select_training_bucket(image_module, image_path: Path, resolutions: list[int], randomizer: random.Random) -> TrainingBucket:
    long_edge = randomizer.choice(resolutions)
    with image_module.open(image_path) as image:
        width, height = image.size
    if width <= 0 or height <= 0:
        return TrainingBucket(width=long_edge, height=long_edge)
    if width >= height:
        bucket_width = long_edge
        bucket_height = _round_to_multiple(long_edge * height / width, TRAINING_SIZE_MULTIPLE)
    else:
        bucket_width = _round_to_multiple(long_edge * width / height, TRAINING_SIZE_MULTIPLE)
        bucket_height = long_edge
    return TrainingBucket(
        width=max(TRAINING_SIZE_MULTIPLE, bucket_width),
        height=max(TRAINING_SIZE_MULTIPLE, bucket_height),
    )


def _round_to_multiple(value: float, multiple: int) -> int:
    return max(multiple, int(round(value / multiple)) * multiple)


def _checkpoint_dir(run_dir: Path, step: int) -> Path:
    return run_dir / "checkpoints" / f"step-{step:06d}"


def _set_shifted_training_timesteps(noise_scheduler, model_input, train_timestep_count: int, device) -> float | None:
    if not _scheduler_config_value(noise_scheduler.config, "use_dynamic_shifting", False):
        noise_scheduler.set_timesteps(train_timestep_count, device=device)
        return None
    image_seq_len = _training_image_seq_len(model_input.shape[-2], model_input.shape[-1])
    shift_mu = _calculate_shift_mu(noise_scheduler.config, image_seq_len)
    noise_scheduler.set_timesteps(train_timestep_count, device=device, mu=shift_mu)
    return shift_mu


def _training_image_seq_len(latent_height: int, latent_width: int) -> int:
    return max(1, int(latent_height) * int(latent_width) // (FLUX_TRAINING_PATCH_SIZE**2))


def _calculate_shift_mu(config, image_seq_len: int) -> float:
    base_seq_len = int(_scheduler_config_value(config, "base_image_seq_len", 256))
    max_seq_len = int(_scheduler_config_value(config, "max_image_seq_len", 4096))
    base_shift = float(_scheduler_config_value(config, "base_shift", 0.5))
    max_shift = float(_scheduler_config_value(config, "max_shift", 1.15))
    if max_seq_len <= base_seq_len:
        return base_shift
    slope = (max_shift - base_shift) / (max_seq_len - base_seq_len)
    return image_seq_len * slope + base_shift - slope * base_seq_len


def _scheduler_config_value(config, key: str, default: Any) -> Any:
    if hasattr(config, "get"):
        return config.get(key, default)
    return getattr(config, key, default)


def _sample_timesteps(torch, noise_scheduler, batch_size: int, device):
    timesteps = noise_scheduler.timesteps.to(device)
    indices = torch.randint(0, len(timesteps), (batch_size,), device=device)
    return timesteps[indices], indices


def _sigmas_for_timesteps(torch, noise_scheduler, timestep_indices, n_dim: int, dtype, device):
    sigmas = noise_scheduler.sigmas.to(device=device, dtype=dtype)[timestep_indices].flatten()
    while len(sigmas.shape) < n_dim:
        sigmas = sigmas.unsqueeze(-1)
    return sigmas


def _append_log(log_path: Path, message: str) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as log_file:
        log_file.write(f"{_utc_now()} {message.rstrip()}\n")


def _read_log_tail(log_path: Path, line_count: int = 40) -> str:
    if not log_path.exists():
        return ""
    return "\n".join(log_path.read_text(encoding="utf-8", errors="replace").splitlines()[-line_count:])


def _update_state(**changes: Any) -> None:
    with _LOCK:
        _STATE.update(changes)


def _state_snapshot_locked() -> dict[str, Any]:
    state = dict(_STATE)
    state["progress"] = dict(_STATE["progress"])
    return state


def _run_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _reset_training_state_for_tests() -> None:
    global _STATE, _STOP_EVENT, _THREAD

    with _LOCK:
        _STATE = _initial_state()
        _STOP_EVENT = None
        _THREAD = None
