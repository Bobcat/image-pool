from __future__ import annotations

import gc
from pathlib import Path
import subprocess
import sys
import uuid
from dataclasses import dataclass, field
from typing import Any, Literal

from app.schemas import ImageInput


class UnknownModelError(RuntimeError):
    pass


class ModelNotLoadedError(RuntimeError):
    pass


class UnsupportedBackendError(RuntimeError):
    pass


def release_torch_cuda_memory(torch_module: Any) -> None:
    gc.collect()
    cuda = getattr(torch_module, "cuda", None)
    if cuda is None or not cuda.is_available():
        return
    cuda.empty_cache()
    ipc_collect = getattr(cuda, "ipc_collect", None)
    if ipc_collect is not None:
        ipc_collect()


def release_loaded_torch_cuda_memory() -> None:
    torch_module = sys.modules.get("torch")
    if torch_module is None:
        gc.collect()
        return
    release_torch_cuda_memory(torch_module)


def query_gpu_memory() -> tuple[list[dict[str, object]], str | None]:
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,name,memory.used,memory.total,memory.free",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except FileNotFoundError:
        return [], "nvidia-smi not found"
    except subprocess.CalledProcessError as exc:
        message = exc.stderr.strip() if isinstance(exc.stderr, str) else ""
        return [], message or "nvidia-smi failed"
    except subprocess.TimeoutExpired:
        return [], "nvidia-smi timed out"

    gpus: list[dict[str, object]] = []
    for raw_line in result.stdout.splitlines():
        parts = [part.strip() for part in raw_line.split(",")]
        if len(parts) < 4:
            continue
        try:
            index = int(parts[0])
            used_mib = int(parts[2])
            total_mib = int(parts[3])
            free_mib = int(parts[4]) if len(parts) > 4 else max(0, total_mib - used_mib)
        except ValueError:
            continue
        gpus.append(
            {
                "index": index,
                "name": parts[1],
                "used_mib": used_mib,
                "total_mib": total_mib,
                "free_mib": free_mib,
                "used_over_total": f"{used_mib}MiB / {total_mib}MiB",
            }
        )
    return gpus, None


def query_primary_gpu_used_mib() -> int | None:
    gpus, _ = query_gpu_memory()
    if not gpus:
        return None
    used = gpus[0].get("used_mib")
    if isinstance(used, int):
        return used
    return None


def estimate_model_artifact_size_mib(model_path: str | None) -> int | None:
    path_value = str(model_path or "").strip()
    if not path_value:
        return None
    path = Path(path_value)
    try:
        if path.is_file():
            total_bytes = path.stat().st_size
        elif path.is_dir():
            total_bytes = 0
            for candidate in path.rglob("*"):
                if candidate.is_file():
                    total_bytes += candidate.stat().st_size
        else:
            return None
    except OSError:
        return None
    if total_bytes <= 0:
        return None
    return max(1, int(total_bytes / (1024 * 1024)))


@dataclass(slots=True)
class GeneratedImagePayload:
    b64_json: str
    mime_type: str = "image/png"
    revised_prompt: str | None = None


@dataclass(slots=True)
class ImageResult:
    images: list[GeneratedImagePayload]
    metrics: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class ImageJob:
    operation: Literal["generation", "edit"]
    model: str
    prompt: str
    size: str
    n: int
    quality: str
    seed: int | None
    metadata: dict[str, Any] = field(default_factory=dict)
    images: tuple[ImageInput, ...] = ()
    request_id: str = field(default_factory=lambda: f"imgreq-{uuid.uuid4().hex}")


@dataclass(slots=True)
class ModelRuntimeState:
    name: str
    backend: str
    enabled: bool
    loaded: bool = False
    loading: bool = False
    target_inflight: int = 1
    loaded_at: float | None = None
    last_error: str | None = None
    observed_vram_mib: int | None = None
    artifact_size_mib: int | None = None
