import unittest

from app.config import load_settings


class ConfigTests(unittest.TestCase):
    def test_default_settings_have_stub_model(self):
        settings = load_settings()

        self.assertIn("stub-image", settings.engine.models)
        self.assertEqual(settings.engine.models["stub-image"].backend, "stub")
        self.assertEqual(settings.engine.models["stub-image"].target_inflight, 1)

    def test_default_settings_have_flux_model_disabled(self):
        settings = load_settings()

        self.assertIn("flux2-klein-4b", settings.engine.models)
        model = settings.engine.models["flux2-klein-4b"]
        self.assertEqual(model.backend, "diffusers_flux2_klein")
        self.assertFalse(model.enabled)
        self.assertEqual(model.target_inflight, 1)
        self.assertEqual(model.model_path, "/home/gunnar/models/FLUX.2-klein-4B")
        self.assertIn("image_edit", model.tasks)

    def test_default_settings_have_flux_9b_model_disabled(self):
        settings = load_settings()

        self.assertIn("flux2-klein-9b", settings.engine.models)
        model = settings.engine.models["flux2-klein-9b"]
        self.assertEqual(model.backend, "diffusers_flux2_klein")
        self.assertFalse(model.enabled)
        self.assertEqual(model.target_inflight, 1)
        self.assertEqual(model.model_path, "/home/gunnar/models/FLUX.2-klein-9B")
        self.assertIn("image_edit", model.tasks)

    def test_default_settings_have_firered_model_disabled(self):
        settings = load_settings()

        self.assertIn("firered-image-edit-1.1-q4-k-m", settings.engine.models)
        model = settings.engine.models["firered-image-edit-1.1-q4-k-m"]
        self.assertEqual(model.backend, "diffusers_firered_gguf")
        self.assertFalse(model.enabled)
        self.assertEqual(model.target_inflight, 1)
        self.assertEqual(model.model_path, "/home/gunnar/models/FireRed-Image-Edit-1.1-Q4_K_M.gguf")
        self.assertEqual(model.base_model_path, "/home/gunnar/models/FireRed-Image-Edit-1.1")
        self.assertEqual(model.max_images, 1)
        self.assertEqual(model.tasks, ("image_edit",))


if __name__ == "__main__":
    unittest.main()
