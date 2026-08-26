import unittest
from unittest.mock import patch, MagicMock
import sys
import os
import json
import base64

# handler.py lives at the repo root; conftest.py puts it on sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import handler


class TestValidateInput(unittest.TestCase):
    def test_valid_input_with_workflow_only(self):
        validated_data, error = handler.validate_input({"workflow": {"key": "value"}})
        self.assertIsNone(error)
        self.assertEqual(
            validated_data,
            {"workflow": {"key": "value"}, "images": None, "comfy_org_api_key": None},
        )

    def test_valid_input_with_workflow_and_images(self):
        input_data = {
            "workflow": {"key": "value"},
            "images": [{"name": "image1.png", "image": "base64string"}],
        }
        validated_data, error = handler.validate_input(input_data)
        self.assertIsNone(error)
        self.assertEqual(validated_data["workflow"], input_data["workflow"])
        self.assertEqual(validated_data["images"], input_data["images"])

    def test_comfy_org_api_key_is_passed_through(self):
        validated_data, error = handler.validate_input(
            {"workflow": {}, "comfy_org_api_key": "comfy-key"}
        )
        self.assertIsNone(error)
        self.assertEqual(validated_data["comfy_org_api_key"], "comfy-key")

    def test_input_missing_workflow(self):
        validated_data, error = handler.validate_input(
            {"images": [{"name": "image1.png", "image": "base64string"}]}
        )
        self.assertIsNotNone(error)
        self.assertEqual(error, "Missing 'workflow' parameter")

    def test_input_with_invalid_images_structure(self):
        validated_data, error = handler.validate_input(
            {"workflow": {"key": "value"}, "images": [{"name": "image1.png"}]}
        )
        self.assertEqual(
            error, "'images' must be a list of objects with 'name' and 'image' keys"
        )

    def test_invalid_json_string_input(self):
        validated_data, error = handler.validate_input("invalid json")
        self.assertEqual(error, "Invalid JSON format in input")

    def test_valid_json_string_input(self):
        validated_data, error = handler.validate_input('{"workflow": {"key": "value"}}')
        self.assertIsNone(error)
        self.assertEqual(validated_data["workflow"], {"key": "value"})

    def test_empty_input(self):
        validated_data, error = handler.validate_input(None)
        self.assertEqual(error, "Please provide input")


class TestCheckServer(unittest.TestCase):
    @patch("handler.requests.get")
    def test_check_server_server_up(self, mock_requests):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_requests.return_value = mock_response

        result = handler.check_server("http://127.0.0.1:8188", 1, 50)
        self.assertTrue(result)

    @patch("handler._is_comfyui_process_alive", return_value=False)
    @patch("handler.requests.get")
    def test_check_server_fails_fast_when_comfyui_process_died(
        self, mock_requests, mock_alive
    ):
        mock_requests.side_effect = handler.requests.RequestException()
        result = handler.check_server("http://127.0.0.1:8188", 2, 1)
        self.assertFalse(result)

    @patch("handler._is_comfyui_process_alive", return_value=None)
    @patch("handler.requests.get")
    def test_check_server_respects_retry_limit_without_pid_file(
        self, mock_requests, mock_alive
    ):
        mock_requests.side_effect = handler.requests.RequestException()
        result = handler.check_server("http://127.0.0.1:8188", 2, 1)
        self.assertFalse(result)


class TestBackgroundProvisioningBoot(unittest.TestCase):
    """Serverless platforms cull workers whose handler is not up within a
    ~10-minute health window, so start.sh boots the handler first and
    provisions in the background. The handler must (a) keep waiting for
    ComfyUI while the provisioning marker exists, and (b) surface a
    provisioning failure as the job's error."""

    @patch("handler.os.path.exists")
    def test_process_reported_alive_while_provisioning(self, mock_exists):
        mock_exists.side_effect = lambda p: p == handler.PROVISIONING_MARKER_FILE
        self.assertTrue(handler._is_comfyui_process_alive())

    @patch("handler.os.path.exists", return_value=False)
    def test_no_marker_falls_back_to_pid_check(self, mock_exists):
        self.assertIsNone(handler._is_comfyui_process_alive())

    def test_job_reports_provisioning_failure(self):
        import tempfile, os as _os
        with tempfile.NamedTemporaryFile("w", delete=False, suffix=".failed") as f:
            f.write("Provisioning failed: CHECKPOINT_URLS 401 (set CIVITAI_TOKEN)")
            failed_path = f.name
        original = handler.PROVISIONING_FAILED_FILE
        handler.PROVISIONING_FAILED_FILE = failed_path
        try:
            result = handler.handler({"id": "j1", "input": {"workflow": {}}})
        finally:
            handler.PROVISIONING_FAILED_FILE = original
            _os.unlink(failed_path)
        self.assertIn("CIVITAI_TOKEN", result["error"])


class TestQueueWorkflow(unittest.TestCase):
    def _mock_post_response(self):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"prompt_id": "123"}
        return mock_response

    @patch("handler.requests.post")
    def test_queue_workflow_posts_prompt_and_client_id(self, mock_post):
        mock_post.return_value = self._mock_post_response()

        result = handler.queue_workflow({"node": {}}, "client-1")

        self.assertEqual(result, {"prompt_id": "123"})
        args, kwargs = mock_post.call_args
        self.assertTrue(args[0].endswith("/prompt"))
        payload = json.loads(kwargs["data"].decode())
        self.assertEqual(payload["prompt"], {"node": {}})
        self.assertEqual(payload["client_id"], "client-1")
        self.assertNotIn("extra_data", payload)

    @patch("handler.requests.post")
    def test_queue_workflow_injects_per_request_comfy_org_api_key(self, mock_post):
        mock_post.return_value = self._mock_post_response()

        handler.queue_workflow({}, "client-1", comfy_org_api_key="req-key")

        payload = json.loads(mock_post.call_args.kwargs["data"].decode())
        self.assertEqual(payload["extra_data"], {"api_key_comfy_org": "req-key"})

    @patch.dict(os.environ, {"COMFY_ORG_API_KEY": "env-key"})
    @patch("handler.requests.post")
    def test_per_request_key_overrides_environment_key(self, mock_post):
        mock_post.return_value = self._mock_post_response()

        handler.queue_workflow({}, "client-1", comfy_org_api_key="req-key")

        payload = json.loads(mock_post.call_args.kwargs["data"].decode())
        self.assertEqual(payload["extra_data"], {"api_key_comfy_org": "req-key"})


class TestGetHistory(unittest.TestCase):
    @patch("handler.requests.get")
    def test_get_history(self, mock_get):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"key": "value"}
        mock_get.return_value = mock_response

        result = handler.get_history("123")

        self.assertEqual(result, {"key": "value"})
        called_url = mock_get.call_args.args[0]
        self.assertTrue(called_url.endswith("/history/123"))


class TestUploadImages(unittest.TestCase):
    def _images(self):
        image_data = base64.b64encode(b"Test Image Data").decode("utf-8")
        return [{"name": "test_image.png", "image": image_data}]

    @patch("handler.requests.post")
    def test_upload_images_successful(self, mock_post):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.raise_for_status.return_value = None
        mock_post.return_value = mock_response

        responses = handler.upload_images(self._images())

        self.assertEqual(responses["status"], "success")

    @patch("handler.requests.post")
    def test_upload_images_failed(self, mock_post):
        mock_response = MagicMock()
        mock_response.status_code = 400
        mock_response.raise_for_status.side_effect = handler.requests.HTTPError(
            "400 Client Error"
        )
        mock_post.return_value = mock_response

        responses = handler.upload_images(self._images())

        self.assertEqual(responses["status"], "error")

    def test_upload_no_images_is_success(self):
        responses = handler.upload_images([])
        self.assertEqual(responses["status"], "success")


if __name__ == "__main__":
    unittest.main()
