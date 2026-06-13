from __future__ import annotations

import subprocess
import time
import uuid
from typing import Any

from app.config import ModelSettings, Settings
from app.engine.common import (
    ImageJob,
    ModelNotLoadedError,
    ModelRuntimeState,
    UnknownModelError,
    UnsupportedBackendError,
    release_loaded_torch_cuda_memory,
)
from app.engine.diffusers_firered_gguf import DiffusersFireRedGgufRuntime
from app.engine.diffusers_flux import DiffusersFlux2KleinRuntime
from app.engine.scheduler import LoadedModelExecutor, RuntimeScheduler
from app.engine.stub import StubImageRuntime
from app.schemas import ImageData, ImageEditRequest, ImageGenerationRequest, ImageResponse


class ImageRouterEngine:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._scheduler = RuntimeScheduler()
        self._states: dict[str, ModelRuntimeState] = {}
        self._runtimes: dict[str, StubImageRuntime | DiffusersFlux2KleinRuntime | DiffusersFireRedGgufRuntime] = {}
        for name, model_settings in settings.engine.models.items():
            self._states[name] = ModelRuntimeState(
                name=name,
                backend=model_settings.backend,
                enabled=model_settings.enabled,
                target_inflight=model_settings.target_inflight,
            )

    async def load_enabled_models(self) -> None:
        for model_name, model_settings in self.settings.engine.models.items():
            if model_settings.enabled:
                await self.load_model(model_name)

    async def close(self) -> None:
        await self._scheduler.close()
        for runtime in self._runtimes.values():
            self._close_runtime(runtime)
        self._runtimes.clear()
        release_loaded_torch_cuda_memory()
        for state in self._states.values():
            state.loaded = False

    async def load_model(self, model_name: str) -> dict[str, Any]:
        model_settings = self._model_settings(model_name)
        state = self._states[model_name]
        if state.loaded:
            return self._state_payload(model_name)
        state.loading = True
        state.last_error = None
        try:
            runtime = self._create_runtime(model_name, model_settings)
            executor = LoadedModelExecutor(
                model_name=model_name,
                complete_fn=runtime.complete,
                target_inflight=model_settings.target_inflight,
            )
            await self._scheduler.register(model_name, executor)
            self._runtimes[model_name] = runtime
            state.loaded = True
            state.loaded_at = time.time()
            state.loading = False
            return self._state_payload(model_name)
        except Exception as exc:
            message = str(exc)
            state.last_error = message
            exc.__traceback__ = None
            exc.__context__ = None
            exc.__cause__ = None
            release_loaded_torch_cuda_memory()
            raise RuntimeError(message) from None
        finally:
            state.loading = False

    async def unload_model(self, model_name: str) -> dict[str, Any]:
        self._model_settings(model_name)
        await self._scheduler.unregister(model_name)
        runtime = self._runtimes.pop(model_name, None)
        self._close_runtime(runtime)
        del runtime
        release_loaded_torch_cuda_memory()
        state = self._states[model_name]
        state.loaded = False
        state.loaded_at = None
        return self._state_payload(model_name)

    async def generate(self, request: ImageGenerationRequest) -> ImageResponse:
        job = ImageJob(
            operation="generation",
            model=request.model,
            prompt=request.prompt,
            size=request.size,
            n=request.n,
            quality=request.quality,
            seed=request.seed,
            metadata=dict(request.metadata),
        )
        return await self._complete(job, "image.generation")

    async def edit(self, request: ImageEditRequest) -> ImageResponse:
        model_settings = self._model_settings(request.model)
        if len(request.images) > model_settings.max_images:
            raise ValueError(f"model accepts at most {model_settings.max_images} input images")
        job = ImageJob(
            operation="edit",
            model=request.model,
            prompt=request.prompt,
            size=request.size,
            n=request.n,
            quality=request.quality,
            seed=request.seed,
            metadata=dict(request.metadata),
            images=tuple(request.images),
        )
        return await self._complete(job, "image.edit")

    async def _complete(self, job: ImageJob, object_name: str) -> ImageResponse:
        model_settings = self._model_settings(job.model)
        if job.n > model_settings.max_output_images:
            raise ValueError(f"model returns at most {model_settings.max_output_images} images")
        started_at = time.perf_counter()
        result = await self._scheduler.complete(job.model, job)
        metrics = dict(result.metrics)
        metrics.setdefault("pool_total_wall_ms", (time.perf_counter() - started_at) * 1000)
        return ImageResponse(
            id=f"img-{uuid.uuid4().hex}",
            object=object_name,
            created=int(time.time()),
            model=job.model,
            data=[ImageData(b64_json=item.b64_json, mime_type=item.mime_type, revised_prompt=item.revised_prompt) for item in result.images],
            metrics=metrics,
        )

    def public_models_payload(self) -> dict[str, Any]:
        data = []
        for model_name in sorted(self.settings.engine.models):
            state = self._states[model_name]
            if state.loaded:
                data.append(self._public_model_payload(model_name))
        return {"object": "list", "data": data}

    def admin_models_payload(self) -> dict[str, Any]:
        return {
            "object": "list",
            "data": [self._state_payload(model_name) for model_name in sorted(self.settings.engine.models)],
        }

    def gpu_memory_payload(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "source": "nvidia-smi",
            "available": False,
            "gpus": [],
            "models": [],
        }
        try:
            completed = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=index,name,memory.total,memory.used,memory.free",
                    "--format=csv,noheader,nounits",
                ],
                check=True,
                capture_output=True,
                text=True,
                timeout=5,
            )
        except Exception as exc:
            payload["error"] = str(exc)
        else:
            payload["available"] = True
            for line in completed.stdout.splitlines():
                index, name, total, used, free = [part.strip() for part in line.split(",", maxsplit=4)]
                payload["gpus"].append(
                    {
                        "index": int(index),
                        "name": name,
                        "memory_total_mib": int(total),
                        "memory_used_mib": int(used),
                        "memory_free_mib": int(free),
                    }
                )
        for model_name, model_settings in sorted(self.settings.engine.models.items()):
            payload["models"].append(
                {
                    "id": model_name,
                    "backend": model_settings.backend,
                    "loaded": self._states[model_name].loaded,
                    "vram_estimate_mib": model_settings.vram_estimate_mib,
                }
            )
        return payload

    def _model_settings(self, model_name: str) -> ModelSettings:
        try:
            return self.settings.engine.models[model_name]
        except KeyError as exc:
            raise UnknownModelError(f"unknown model: {model_name}") from exc

    def _create_runtime(self, model_name: str, model_settings: ModelSettings) -> StubImageRuntime | DiffusersFlux2KleinRuntime | DiffusersFireRedGgufRuntime:
        if model_settings.backend == "stub":
            return StubImageRuntime(model_name, model_settings)
        if model_settings.backend == "diffusers_flux2_klein":
            return DiffusersFlux2KleinRuntime(model_name, model_settings)
        if model_settings.backend == "diffusers_firered_gguf":
            return DiffusersFireRedGgufRuntime(model_name, model_settings)
        raise UnsupportedBackendError(f"unsupported backend: {model_settings.backend}")

    def _close_runtime(self, runtime: object | None) -> None:
        close = getattr(runtime, "close", None)
        if close is not None:
            close()

    def _public_model_payload(self, model_name: str) -> dict[str, Any]:
        model_settings = self._model_settings(model_name)
        return {
            "id": model_name,
            "object": "model",
            "owned_by": "image-pool",
            "backend": model_settings.backend,
            "capabilities": self._capabilities_payload(model_settings),
        }

    def _state_payload(self, model_name: str) -> dict[str, Any]:
        model_settings = self._model_settings(model_name)
        state = self._states[model_name]
        scheduler_state = self._scheduler.snapshot(model_name) or {
            "target_inflight": state.target_inflight,
            "inflight": 0,
            "queued": 0,
        }
        return {
            "id": model_name,
            "backend": model_settings.backend,
            "enabled": state.enabled,
            "loaded": state.loaded,
            "loading": state.loading,
            "loaded_at": state.loaded_at,
            "last_error": state.last_error,
            "scheduler": scheduler_state,
            "capabilities": self._capabilities_payload(model_settings),
            "model_path": model_settings.model_path,
            "base_model_path": model_settings.base_model_path,
            "vram_estimate_mib": model_settings.vram_estimate_mib,
        }

    def _capabilities_payload(self, model_settings: ModelSettings) -> dict[str, Any]:
        return {
            "input_modalities": list(model_settings.modalities),
            "output_modalities": list(model_settings.output_modalities),
            "tasks": list(model_settings.tasks),
            "max_images": model_settings.max_images,
            "max_output_images": model_settings.max_output_images,
        }
