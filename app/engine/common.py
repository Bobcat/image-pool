from __future__ import annotations

import gc
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
