import base64
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient
from PIL import Image

from app import loras as loras_module
from app import training as training_module
from app.engine.diffusers_sdxl import _lora_request_from_metadata as _sdxl_lora_request_from_metadata
from app.engine.diffusers_sdxl import _metadata_sampler
from app.engine.diffusers_flux import _lora_request_from_metadata
from app.engine.diffusers_z_image import _lora_request_from_metadata as _z_image_lora_request_from_metadata
from app.main import create_app


def _write_test_lora(path: Path, *, metadata: dict[str, str] | None = None) -> None:
    import torch
    from safetensors.torch import save_file

    save_file(
        {"transformer.single_transformer_blocks.0.attn.to_out.lora_A.weight": torch.zeros((1, 1))},
        str(path),
        metadata=metadata or {"ss_trigger_words": "TEST_TOKEN"},
    )


def _write_sdxl_test_lora(path: Path, *, metadata: dict[str, str] | None = None) -> None:
    import torch
    from safetensors.torch import save_file

    save_file(
        {"lora_unet_down_blocks_0_attentions_0_transformer_blocks_0_attn1_to_q.lora_A.weight": torch.zeros((1, 1))},
        str(path),
        metadata=metadata or {"ss_trigger_words": "SDXL_TOKEN"},
    )


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

    def test_sdxl_lora_metadata_resolves_request(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            lora_path = Path(tmpdir) / "adapter.safetensors"
            lora_path.write_bytes(b"lora")

            payload = _sdxl_lora_request_from_metadata(
                {
                    "lora_id": "sdxl-lora",
                    "lora_path": str(lora_path),
                    "lora_scale": 1.1,
                }
            )

        self.assertEqual(payload["id"], "sdxl-lora")
        self.assertEqual(payload["path"], str(lora_path.resolve()))
        self.assertEqual(payload["scale"], 1.1)

    def test_sdxl_sampler_metadata_allows_known_values_only(self):
        self.assertEqual(_metadata_sampler({}, default="euler"), "euler")
        self.assertEqual(_metadata_sampler({"sampler": "euler_a"}, default="euler"), "euler_a")
        self.assertEqual(_metadata_sampler({"sampler": "dpmpp_2m"}, default="euler"), "dpmpp_2m")
        self.assertEqual(_metadata_sampler({"sampler": "lcm"}, default="euler"), "lcm")
        self.assertEqual(_metadata_sampler({"sampler": "flowmatch_euler"}, default="euler"), "euler")

    def test_public_models_report_generation_parameters(self):
        with TestClient(create_app()) as client:
            response = client.get("/v1/models")

        self.assertEqual(response.status_code, 200)
        model = next(item for item in response.json()["data"] if item["id"] == "stub-image")
        self.assertEqual(model["generation_parameters"]["size"]["default"], "512x512")
        self.assertEqual(model["generation_parameters"]["n"]["maximum"], 4)
        self.assertEqual(model["edit_parameters"]["size"]["default"], "512x512")

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
                        "recommended_guidance": 4.0,
                        "generation_parameters": {
                          "steps": {"kind": "integer", "target": "metadata", "default": 50}
                        },
                        "edit_parameters": {
                          "strength": {"kind": "number", "target": "metadata", "default": 0.35}
                        }
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
        self.assertEqual(model["generation_parameters"]["steps"]["default"], 50)
        self.assertEqual(model["edit_parameters"]["strength"]["default"], 0.35)

    def test_lora_inspect_reports_family_and_model_suggestions(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            source_path = Path(tmpdir) / "external.safetensors"
            _write_test_lora(source_path)
            settings_path = Path(tmpdir) / "settings.json"
            settings_path.write_text(
                json.dumps(
                    {
                        "engine": {
                            "models": {
                                "flux-test": {
                                    "backend": "diffusers_flux2_klein",
                                    "enabled": False,
                                    "model_path": "/tmp/flux-test",
                                    "generation_parameters": {
                                        "lora_scale": {"kind": "number", "target": "metadata", "default": 0.35}
                                    },
                                },
                                "sdxl-test": {
                                    "backend": "diffusers_sdxl",
                                    "enabled": False,
                                    "model_path": "/tmp/sdxl-test",
                                },
                            }
                        }
                    }
                ),
                encoding="utf-8",
            )

            with TestClient(create_app(settings_path)) as client:
                response = client.post("/v1/admin/loras/inspect", json={"source_path": str(source_path)})

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["family_guess"], "flux2-klein")
        self.assertEqual(payload["trigger_words"], ["TEST_TOKEN"])
        self.assertEqual(payload["compatible_model_suggestions"], ["flux-test"])
        self.assertEqual(payload["default_strength_suggestion"], 0.35)
        self.assertEqual(payload["detected_modules"], ["transformer"])
        self.assertEqual(payload["key_count"], 1)

    def test_lora_inspect_reports_sdxl_model_suggestions(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            source_path = Path(tmpdir) / "sdxl.safetensors"
            _write_sdxl_test_lora(source_path)
            settings_path = Path(tmpdir) / "settings.json"
            settings_path.write_text(
                json.dumps(
                    {
                        "engine": {
                            "models": {
                                "sdxl-test": {
                                    "backend": "diffusers_sdxl",
                                    "enabled": False,
                                    "model_path": "/tmp/sdxl-test",
                                    "generation_parameters": {
                                        "lora_scale": {"kind": "number", "target": "metadata", "default": 1.0}
                                    },
                                }
                            }
                        }
                    }
                ),
                encoding="utf-8",
            )

            with TestClient(create_app(settings_path)) as client:
                response = client.post("/v1/admin/loras/inspect", json={"source_path": str(source_path)})

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["family_guess"], "sdxl")
        self.assertEqual(payload["trigger_words"], ["SDXL_TOKEN"])
        self.assertEqual(payload["compatible_model_suggestions"], ["sdxl-test"])
        self.assertEqual(payload["default_strength_suggestion"], 1.0)
        self.assertEqual(payload["detected_modules"], ["transformer", "unet"])

    def test_lora_import_copies_weights_and_lists_record(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            source_path = root / "external.safetensors"
            _write_test_lora(source_path)
            settings_path = root / "settings.json"
            settings_path.write_text(
                json.dumps(
                    {
                        "engine": {
                            "models": {
                                "flux-test": {
                                    "backend": "diffusers_flux2_klein",
                                    "enabled": False,
                                    "model_path": "/tmp/flux-test",
                                    "generation_parameters": {
                                        "lora_scale": {"kind": "number", "target": "metadata", "default": 0.35}
                                    },
                                }
                            }
                        }
                    }
                ),
                encoding="utf-8",
            )
            imported_root = root / "imported"

            with patch.object(loras_module, "IMPORTED_LORAS_ROOT", imported_root):
                with TestClient(create_app(settings_path)) as client:
                    import_response = client.post(
                        "/v1/admin/loras/import",
                        json={
                            "source_path": str(source_path),
                            "name": "External Test",
                            "family": "flux2-klein",
                            "compatible_models": ["flux-test"],
                            "trained_on_model_id": "flux-test",
                            "trigger_words": ["TEST_TOKEN"],
                            "default_strength": 0.5,
                            "description": "Imported test LoRA.",
                            "source_url": "https://example.test/lora",
                        },
                    )
                    list_response = client.get("/v1/admin/loras")
                    imported_file_exists = (imported_root / "external-test" / "adapter.safetensors").is_file()
                    metadata = json.loads(
                        (imported_root / "external-test" / "metadata.json").read_text(encoding="utf-8")
                    )

        self.assertEqual(import_response.status_code, 200)
        imported = import_response.json()["lora"]
        self.assertEqual(imported["id"], "imported/external-test")
        self.assertEqual(imported["name"], "External Test")
        self.assertEqual(imported["family"], "flux2-klein")
        self.assertEqual(imported["compatible_models"], ["flux-test"])
        self.assertEqual(imported["trigger_words"], ["TEST_TOKEN"])
        self.assertEqual(imported["default_strength"], 0.5)
        self.assertTrue(imported_file_exists)
        self.assertEqual(metadata["description"], "Imported test LoRA.")
        self.assertEqual(list_response.status_code, 200)
        self.assertEqual(list_response.json()["data"][0]["id"], "imported/external-test")

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
