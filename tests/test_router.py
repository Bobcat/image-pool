import unittest
from unittest import mock

from app.config import EngineSettings, ModelSettings, ServiceSettings, Settings
from app.engine.common import GeneratedImagePayload, ImageJob, ImageResult
from app.engine import router as router_module
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

    async def test_load_records_observed_vram_delta(self):
        settings = Settings(
            service=ServiceSettings(),
            engine=EngineSettings(
                models={
                    "test-image": ModelSettings(
                        backend="stub",
                        enabled=False,
                        modalities=("text", "image"),
                        tasks=("image_generation", "image_edit"),
                        vram_estimate_mib=9999,
                    )
                }
            ),
        )
        engine = ImageRouterEngine(settings)
        runtime = ClosableRuntime()

        with (
            mock.patch.object(engine, "_create_runtime", return_value=runtime),
            mock.patch("app.engine.router.query_primary_gpu_used_mib", side_effect=[1000, 1245]),
        ):
            payload = await engine.load_model("test-image")

        self.assertTrue(payload["loaded"])
        self.assertEqual(payload["vram_estimate_mib"], 245)
        self.assertEqual(payload["vram_estimate_source"], "observed_load_delta")

        with mock.patch("app.engine.router.query_gpu_memory", return_value=([], None)):
            gpu_payload = engine.gpu_memory_payload()

        self.assertEqual(gpu_payload["models"][0]["name"], "test-image")
        self.assertEqual(gpu_payload["models"][0]["vram_estimate_mib"], 245)
        self.assertEqual(gpu_payload["models"][0]["vram_estimate_source"], "observed_load_delta")

    def test_observed_vram_delta_ignores_negative_or_empty_samples(self):
        self.assertIsNone(router_module._observed_vram_delta_mib(None, 1200))
        self.assertIsNone(router_module._observed_vram_delta_mib(1200, None))
        self.assertIsNone(router_module._observed_vram_delta_mib(1200, 1100))
        self.assertIsNone(router_module._observed_vram_delta_mib(1200, 1200))
        self.assertEqual(router_module._observed_vram_delta_mib(1200, 1456), 256)


if __name__ == "__main__":
    unittest.main()
