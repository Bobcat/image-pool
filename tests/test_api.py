import base64
import unittest

from fastapi.testclient import TestClient

from app.main import create_app


class ApiTests(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
