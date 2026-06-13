import unittest
from unittest import mock

from app.config import EngineSettings, ModelSettings, ServiceSettings, Settings
from app.engine.common import GeneratedImagePayload, ImageJob, ImageResult
from app.engine.router import ImageRouterEngine


class ClosableRuntime:
    def __init__(self):
        self.closed = False

    def close(self):
        self.closed = True

    def complete(self, job: ImageJob) -> ImageResult:
        return ImageResult(images=[GeneratedImagePayload(b64_json="")])


class RouterTests(unittest.IsolatedAsyncioTestCase):
    async def test_unload_closes_runtime(self):
        settings = Settings(
            service=ServiceSettings(),
            engine=EngineSettings(
                models={
                    "test-image": ModelSettings(
                        backend="stub",
                        enabled=False,
                        modalities=("text", "image"),
                        tasks=("image_generation", "image_edit"),
                    )
                }
            ),
        )
        engine = ImageRouterEngine(settings)
        runtime = ClosableRuntime()

        with (
            mock.patch.object(engine, "_create_runtime", return_value=runtime),
            mock.patch("app.engine.router.release_loaded_torch_cuda_memory") as release_memory,
        ):
            await engine.load_model("test-image")
            payload = await engine.unload_model("test-image")

        self.assertTrue(runtime.closed)
        self.assertFalse(payload["loaded"])
        release_memory.assert_called_once()


if __name__ == "__main__":
    unittest.main()
