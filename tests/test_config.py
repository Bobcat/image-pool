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

    def test_default_settings_have_sdxl_model_disabled(self):
        settings = load_settings()

        self.assertIn("sdxl-base-1.0", settings.engine.models)
        model = settings.engine.models["sdxl-base-1.0"]
        self.assertEqual(model.backend, "diffusers_sdxl")
        self.assertFalse(model.enabled)
        self.assertEqual(model.target_inflight, 1)
        self.assertEqual(model.model_path, "/home/gunnar/models/stable-diffusion-xl-base-1.0")
        self.assertEqual(model.max_images, 1)
        self.assertEqual(model.tasks, ("image_generation", "image_edit"))
        self.assertEqual(model.modalities, ("text", "image"))
        self.assertEqual(model.recommended_steps, 30)
        self.assertEqual(model.recommended_guidance, 5.0)

    def test_default_settings_have_z_image_turbo_model_disabled(self):
        settings = load_settings()

        self.assertIn("z-image-turbo", settings.engine.models)
        model = settings.engine.models["z-image-turbo"]
        self.assertEqual(model.backend, "diffusers_z_image")
        self.assertFalse(model.enabled)
        self.assertEqual(model.target_inflight, 1)
        self.assertEqual(model.model_path, "/home/gunnar/models/Z-Image-Turbo")
        self.assertEqual(model.max_images, 1)
        self.assertEqual(model.tasks, ("image_generation", "image_edit"))
        self.assertEqual(model.modalities, ("text", "image"))
        self.assertEqual(model.recommended_steps, 9)
        self.assertEqual(model.recommended_guidance, 0.0)

    def test_default_settings_have_z_image_base_model_disabled(self):
        settings = load_settings()

        self.assertIn("z-image-base", settings.engine.models)
        model = settings.engine.models["z-image-base"]
        self.assertEqual(model.backend, "diffusers_z_image")
        self.assertFalse(model.enabled)
        self.assertEqual(model.target_inflight, 1)
        self.assertEqual(model.model_path, "/home/gunnar/models/Z-Image")
        self.assertEqual(model.max_images, 1)
        self.assertEqual(model.tasks, ("image_generation", "image_edit"))
        self.assertEqual(model.modalities, ("text", "image"))
        self.assertEqual(model.recommended_steps, 50)
        self.assertEqual(model.recommended_guidance, 5.0)


if __name__ == "__main__":
    unittest.main()
