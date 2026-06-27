import base64
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient
from PIL import Image

from app import training as training_module
from app.engine.diffusers_flux import _lora_request_from_metadata
from app.engine.diffusers_z_image import _lora_request_from_metadata as _z_image_lora_request_from_metadata
from app.main import create_app


class ApiTests(unittest.TestCase):
    def setUp(self) -> None:
        training_module._reset_training_state_for_tests()

    def tearDown(self) -> None:
        training_module._reset_training_state_for_tests()

    def test_stub_generation_returns_png(self):
        with TestClient(create_app()) as client:
            response = client.post(
                "/v1/images/generations",
                json={"model": "stub-image", "prompt": "small test", "size": "64x64"},
            )

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["object"], "image.generation")
        self.assertEqual(payload["model"], "stub-image")
        png = base64.b64decode(payload["data"][0]["b64_json"])
        self.assertTrue(png.startswith(b"\x89PNG\r\n\x1a\n"))
        self.assertEqual(payload["metrics"]["backend"], "stub")

    def test_stub_edit_requires_input_image(self):
        with TestClient(create_app()) as client:
            response = client.post(
                "/v1/images/edits",
                json={"model": "stub-image", "prompt": "edit this", "size": "64x64", "images": []},
            )

        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.json()["error"]["type"], "bad_request")

    def test_unloaded_model_rejects_generation(self):
        with TestClient(create_app()) as client:
            unload_response = client.post("/v1/admin/models/stub-image/unload")
            response = client.post(
                "/v1/images/generations",
                json={"model": "stub-image", "prompt": "small test", "size": "64x64"},
            )

        self.assertEqual(unload_response.status_code, 200)
        self.assertEqual(response.status_code, 409)

    def test_flux_lora_metadata_resolves_request(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            lora_path = Path(tmpdir) / "pytorch_lora_weights.safetensors"
            lora_path.write_bytes(b"lora")

            payload = _lora_request_from_metadata(
                {
                    "lora_id": "test-lora",
                    "lora_path": str(lora_path),
                    "lora_scale": 0.8,
                }
            )

        self.assertEqual(payload["id"], "test-lora")
        self.assertEqual(payload["path"], str(lora_path.resolve()))
        self.assertEqual(payload["scale"], 0.8)

    def test_flux_lora_metadata_ignores_missing_or_zero_scale(self):
        self.assertIsNone(_lora_request_from_metadata({}))
        self.assertIsNone(
            _lora_request_from_metadata(
                {
                    "lora_path": "/missing/file.safetensors",
                    "lora_scale": 0,
                }
            )
        )

    def test_z_image_lora_metadata_resolves_request(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            lora_path = Path(tmpdir) / "pytorch_lora_weights.safetensors"
            lora_path.write_bytes(b"lora")

            payload = _z_image_lora_request_from_metadata(
                {
                    "lora_id": "z-lora",
                    "lora_path": str(lora_path),
                    "lora_scale": 0.65,
                }
            )

        self.assertEqual(payload["id"], "z-lora")
        self.assertEqual(payload["path"], str(lora_path.resolve()))
        self.assertEqual(payload["scale"], 0.65)

    def test_training_status_reports_backend(self):
        with TestClient(create_app()) as client:
            response = client.get("/v1/training/flux-lora")

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["backend"]["id"], "diffusers_flux2_lora")
        self.assertTrue(payload["backend"]["implemented"])
        self.assertEqual(payload["run"]["status"], "idle")

    def test_z_image_training_status_reports_backend(self):
        with TestClient(create_app()) as client:
            response = client.get("/v1/training/z-image-lora")

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["backend"]["id"], "diffusers_z_image_lora")
        self.assertTrue(payload["backend"]["implemented"])
        self.assertEqual(payload["run"]["status"], "idle")

    def test_admin_models_report_recommended_generation_defaults(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            settings_path = Path(tmpdir) / "settings.json"
            settings_path.write_text(
                """
                {
                  "engine": {
                    "models": {
                      "flux-test": {
                        "backend": "diffusers_flux2_klein",
                        "enabled": false,
                        "model_path": "/tmp/flux-test",
                        "recommended_steps": 50,
                        "recommended_guidance": 4.0
                      }
                    }
                  }
                }
                """,
                encoding="utf-8",
            )

            with TestClient(create_app(settings_path)) as client:
                response = client.get("/v1/admin/models")

        self.assertEqual(response.status_code, 200)
        model = response.json()["data"][0]
        self.assertEqual(model["id"], "flux-test")
        self.assertEqual(model["recommended_steps"], 50)
        self.assertEqual(model["recommended_guidance"], 4.0)

    def test_training_bucket_preserves_image_aspect_ratio(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            image_path = Path(tmpdir) / "wide.png"
            Image.new("RGB", (2752, 1536), "white").save(image_path)

            bucket = training_module._select_training_bucket(
                Image,
                image_path,
                [1024],
                training_module.random.Random(0),
            )

        self.assertEqual(bucket.width, 1024)
        self.assertEqual(bucket.height, 576)

    def test_training_resolutions_uses_requested_buckets(self):
        self.assertEqual(
            training_module._training_resolutions([256, 512, 768, 1024, 1280, 1536]),
            [256, 512, 768, 1024, 1280, 1536],
        )

    def test_training_checkpoint_dir_uses_step_label(self):
        self.assertEqual(
            training_module._checkpoint_dir(Path("/tmp/run"), 500),
            Path("/tmp/run/checkpoints/step-000500"),
        )

    def test_training_shift_mu_uses_flux_scheduler_config(self):
        config = {
            "base_image_seq_len": 256,
            "max_image_seq_len": 4096,
            "base_shift": 0.5,
            "max_shift": 1.15,
        }

        image_seq_len = training_module._training_image_seq_len(128, 72)
        shift_mu = training_module._calculate_shift_mu(config, image_seq_len)

        self.assertEqual(image_seq_len, 2304)
        self.assertAlmostEqual(shift_mu, 0.8466666667)

    def test_training_start_requires_captioned_dataset(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            with TestClient(create_app()) as client:
                response = client.post(
                    "/v1/training/flux-lora",
                    json={
                        "model": "flux2-klein-4b",
                        "dataset_path": tmpdir,
                        "output_path": str(Path(tmpdir) / "output"),
                    },
                )

        self.assertEqual(response.status_code, 409)
        self.assertEqual(response.json()["detail"]["error"], "training_dataset_not_ready")

    def test_training_start_reports_unavailable_dependency(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            dataset_path = Path(tmpdir) / "dataset"
            output_path = Path(tmpdir) / "output"
            model_path = Path(tmpdir) / "model"
            dataset_path.mkdir()
            model_path.mkdir()
            (dataset_path / "example.png").write_bytes(b"image")
            (dataset_path / "example.txt").write_text("GFX_IMPR5N. Caption.\n", encoding="utf-8")
            settings_path = Path(tmpdir) / "settings.json"
            settings_path.write_text(
                """
                {
                  "engine": {
                    "models": {
                      "flux-test": {
                        "backend": "diffusers_flux2_klein",
                        "enabled": false,
                        "model_path": "%s",
                        "modalities": ["text", "image"],
                        "tasks": ["image_generation", "image_edit"]
                      }
                    }
                  }
                }
                """
                % str(model_path),
                encoding="utf-8",
            )

            with (
                patch(
                    "app.training._backend_status",
                    return_value={
                        "id": "diffusers_flux2_lora",
                        "implemented": True,
                        "available": False,
                        "missing_dependencies": ["peft"],
                        "message": "Missing training dependency: peft.",
                    },
                ),
                TestClient(create_app(settings_path)) as client,
            ):
                response = client.post(
                    "/v1/training/flux-lora",
                    json={
                        "model": "flux-test",
                        "dataset_path": str(dataset_path),
                        "output_path": str(output_path),
                    },
                )

        self.assertEqual(response.status_code, 501)
        payload = response.json()
        self.assertEqual(payload["detail"]["error"], "training_backend_unavailable")
        self.assertTrue(payload["detail"]["preflight"]["dataset"]["ready"])
        self.assertTrue(payload["detail"]["preflight"]["model"]["ready"])

    def test_training_start_launches_internal_trainer(self):
        def fake_start_thread(_request, _model_payload, _run_dir, _stop_event):
            training_module._update_state(
                status="completed",
                completed_at=training_module._utc_now(),
                returncode=0,
                message="Fake training completed.",
            )
            return None

        with tempfile.TemporaryDirectory() as tmpdir:
            dataset_path = Path(tmpdir) / "dataset"
            output_path = Path(tmpdir) / "output"
            model_path = Path(tmpdir) / "model"
            dataset_path.mkdir()
            model_path.mkdir()
            (dataset_path / "example.png").write_bytes(b"image")
            (dataset_path / "example.txt").write_text("GFX_IMPR5N. Caption.\n", encoding="utf-8")
            settings_path = Path(tmpdir) / "settings.json"
            settings_path.write_text(
                """
                {
                  "engine": {
                    "models": {
                      "flux-test": {
                        "backend": "diffusers_flux2_klein",
                        "enabled": false,
                        "model_path": "%s",
                        "modalities": ["text", "image"],
                        "tasks": ["image_generation", "image_edit"]
                      }
                    }
                  }
                }
                """
                % str(model_path),
                encoding="utf-8",
            )

            with (
                patch(
                    "app.training._backend_status",
                    return_value={
                        "id": "diffusers_flux2_lora",
                        "implemented": True,
                        "available": True,
                        "missing_dependencies": [],
                        "message": "Flux LoRA trainer is available.",
                    },
                ),
                patch("app.training._start_training_thread", side_effect=fake_start_thread),
                TestClient(create_app(settings_path)) as client,
            ):
                response = client.post(
                    "/v1/training/flux-lora",
                    json={
                        "model": "flux-test",
                        "dataset_path": str(dataset_path),
                        "output_path": str(output_path),
                    },
                )

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["run"]["status"], "completed")
        self.assertTrue(payload["run"]["output_path"].startswith(str(output_path)))
        self.assertEqual(payload["run"]["progress"]["steps"], 3000)

    def test_z_image_training_start_launches_internal_trainer(self):
        def fake_start_thread(_request, _model_payload, _run_dir, _stop_event):
            training_module._update_state(
                status="completed",
                completed_at=training_module._utc_now(),
                returncode=0,
                message="Fake Z-Image training completed.",
            )
            return None

        with tempfile.TemporaryDirectory() as tmpdir:
            dataset_path = Path(tmpdir) / "dataset"
            output_path = Path(tmpdir) / "output"
            model_path = Path(tmpdir) / "model"
            dataset_path.mkdir()
            model_path.mkdir()
            (dataset_path / "example.png").write_bytes(b"image")
            (dataset_path / "example.txt").write_text("GFX_IMPR5N. Caption.\n", encoding="utf-8")
            settings_path = Path(tmpdir) / "settings.json"
            settings_path.write_text(
                """
                {
                  "engine": {
                    "models": {
                      "z-image-test": {
                        "backend": "diffusers_z_image",
                        "enabled": false,
                        "model_path": "%s",
                        "modalities": ["text", "image"],
                        "tasks": ["image_generation", "image_edit"]
                      }
                    }
                  }
                }
                """
                % str(model_path),
                encoding="utf-8",
            )

            with (
                patch(
                    "app.training._backend_status",
                    return_value={
                        "id": "diffusers_z_image_lora",
                        "implemented": True,
                        "available": True,
                        "missing_dependencies": [],
                        "message": "Z-Image LoRA trainer is available.",
                    },
                ),
                patch("app.training._start_z_image_training_thread", side_effect=fake_start_thread),
                TestClient(create_app(settings_path)) as client,
            ):
                response = client.post(
                    "/v1/training/z-image-lora",
                    json={
                        "model": "z-image-test",
                        "dataset_path": str(dataset_path),
                        "output_path": str(output_path),
                        "steps": 10,
                        "checkpoint_interval": 5,
                    },
                )

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["run"]["status"], "completed")
        self.assertTrue(payload["run"]["output_path"].startswith(str(output_path)))
        self.assertEqual(payload["run"]["progress"]["steps"], 10)
        self.assertEqual(payload["backend"]["id"], "diffusers_z_image_lora")


if __name__ == "__main__":
    unittest.main()
