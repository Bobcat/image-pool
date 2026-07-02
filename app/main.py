from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import JSONResponse

from app.config import load_settings
from app.engine.common import ModelNotLoadedError, UnknownModelError, UnsupportedBackendError
from app.engine.router import ImageRouterEngine
from app.loras import LoraImportRequest, LoraInspectRequest, import_lora, inspect_lora, list_loras
from app.schemas import FluxLoraTrainingStartRequest, ImageEditRequest, ImageGenerationRequest, ZImageLoraTrainingStartRequest
from app.training import start_flux_lora_training, start_z_image_lora_training, stop_training, training_status


def create_app(settings_path: str | Path | None = None) -> FastAPI:
    settings = load_settings(settings_path)
    engine = ImageRouterEngine(settings)

    @asynccontextmanager
    async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
        await engine.load_enabled_models()
        try:
            yield
        finally:
            await engine.close()

    app = FastAPI(title="image-pool", version="0.1.0", lifespan=lifespan)
    app.state.engine = engine

    @app.exception_handler(UnknownModelError)
    async def _unknown_model_handler(_request, exc: UnknownModelError) -> JSONResponse:
        return JSONResponse(status_code=404, content={"error": {"message": str(exc), "type": "unknown_model"}})

    @app.exception_handler(ModelNotLoadedError)
    async def _model_not_loaded_handler(_request, exc: ModelNotLoadedError) -> JSONResponse:
        return JSONResponse(status_code=409, content={"error": {"message": str(exc), "type": "model_not_loaded"}})

    @app.exception_handler(UnsupportedBackendError)
    async def _unsupported_backend_handler(_request, exc: UnsupportedBackendError) -> JSONResponse:
        return JSONResponse(status_code=400, content={"error": {"message": str(exc), "type": "unsupported_backend"}})

    @app.exception_handler(ValueError)
    async def _value_error_handler(_request, exc: ValueError) -> JSONResponse:
        return JSONResponse(status_code=400, content={"error": {"message": str(exc), "type": "bad_request"}})

    @app.get("/healthz")
    async def healthz() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/v1/models")
    async def list_models() -> dict:
        return engine.public_models_payload()

    @app.get("/v1/admin/models")
    async def admin_models() -> dict:
        return engine.admin_models_payload()

    @app.get("/v1/admin/gpu-memory")
    async def gpu_memory() -> dict:
        return engine.gpu_memory_payload()

    @app.get("/v1/admin/loras")
    async def admin_loras() -> dict:
        return list_loras()

    @app.post("/v1/admin/loras/inspect")
    async def admin_lora_inspect(request: LoraInspectRequest) -> dict:
        return inspect_lora(engine.settings, request)

    @app.post("/v1/admin/loras/import")
    async def admin_lora_import(request: LoraImportRequest) -> dict:
        return import_lora(engine.settings, request)

    @app.post("/v1/admin/models/{model_name:path}/load")
    async def load_model(model_name: str) -> dict:
        return await engine.load_model(model_name)

    @app.post("/v1/admin/models/{model_name:path}/unload")
    async def unload_model(model_name: str) -> dict:
        return await engine.unload_model(model_name)

    @app.post("/v1/images/generations")
    async def image_generations(request: ImageGenerationRequest) -> dict:
        return (await engine.generate(request)).model_dump()

    @app.post("/v1/images/edits")
    async def image_edits(request: ImageEditRequest) -> dict:
        return (await engine.edit(request)).model_dump()

    @app.get("/v1/training/flux-lora")
    async def get_flux_lora_training() -> dict:
        return training_status("flux")

    @app.post("/v1/training/flux-lora")
    async def start_flux_lora_training_run(request: FluxLoraTrainingStartRequest) -> dict:
        return start_flux_lora_training(engine, request)

    @app.post("/v1/training/flux-lora/stop")
    async def stop_flux_lora_training_run() -> dict:
        return stop_training("flux")

    @app.get("/v1/training/z-image-lora")
    async def get_z_image_lora_training() -> dict:
        return training_status("z-image")

    @app.post("/v1/training/z-image-lora")
    async def start_z_image_lora_training_run(request: ZImageLoraTrainingStartRequest) -> dict:
        return start_z_image_lora_training(engine, request)

    @app.post("/v1/training/z-image-lora/stop")
    async def stop_z_image_lora_training_run() -> dict:
        return stop_training("z-image")

    return app


app = create_app()
