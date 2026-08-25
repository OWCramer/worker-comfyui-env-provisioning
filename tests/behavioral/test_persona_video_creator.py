"""Persona: Vera — runs video generation workflows (Wan, Hunyuan, LTX, SVD).

Vera's workflows end in video-save nodes, and those nodes don't agree on how
to report their outputs: core SaveVideo reports mp4s under "images",
VHS_VideoCombine reports under "gifs" (even for mp4 files), and some nodes
use "videos". Whatever the node, Vera must get her video back — in the
response's "videos" list — and image outputs must keep working exactly as
before for every existing user.

These tests exercise handler.collect_output_media plus the response-assembly
contract, with the ComfyUI /view fetch and S3 upload mocked.
"""

import base64
import os
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import handler

MP4_BYTES = b"\x00\x00\x00\x20ftypisom-fake-mp4"
PNG_BYTES = b"\x89PNG-fake-png"
WEBP_BYTES = b"RIFF-fake-webp"


def entry(filename, subfolder="", file_type="output"):
    return {"filename": filename, "subfolder": subfolder, "type": file_type}


@pytest.fixture
def fetch_bytes():
    """Mock the /view fetch; maps filename extension to fake bytes."""

    def fake_get_image_data(filename, subfolder, file_type):
        if filename.endswith(".mp4") or filename.endswith(".webm"):
            return MP4_BYTES
        if filename.endswith(".webp"):
            return WEBP_BYTES
        return PNG_BYTES

    with patch.object(handler, "get_image_data", side_effect=fake_get_image_data):
        yield fake_get_image_data


class TestVideoOutputKeys:
    """Every reporting style used by real video-save nodes must work."""

    def test_vhs_videocombine_gifs_key_yields_a_video(self, fetch_bytes):
        outputs = {"12": {"gifs": [entry("wan_video_00001.mp4")]}}
        media, errors = handler.collect_output_media(outputs, "job-1")
        assert errors == []
        assert len(media) == 1
        assert media[0]["kind"] == "video"
        assert media[0]["data"] == base64.b64encode(MP4_BYTES).decode()

    def test_videos_key_yields_a_video(self, fetch_bytes):
        outputs = {"7": {"videos": [entry("ltx_output.webm")]}}
        media, errors = handler.collect_output_media(outputs, "job-1")
        assert [m["kind"] for m in media] == ["video"]

    def test_core_savevideo_mp4_under_images_key_is_classified_as_video(
        self, fetch_bytes
    ):
        """Core SaveVideo reports its mp4 under 'images' — extension wins."""
        outputs = {"9": {"images": [entry("ComfyUI_00001_.mp4")]}}
        media, errors = handler.collect_output_media(outputs, "job-1")
        assert media[0]["kind"] == "video"

    def test_animated_webp_stays_an_image(self, fetch_bytes):
        """SaveAnimatedWEBP produces .webp files — those remain images."""
        outputs = {"5": {"images": [entry("animation.webp")]}}
        media, errors = handler.collect_output_media(outputs, "job-1")
        assert media[0]["kind"] == "image"


class TestImageRegression:
    """Existing image workflows must behave exactly as before."""

    def test_plain_image_workflow_is_unchanged(self, fetch_bytes):
        outputs = {"9": {"images": [entry("ComfyUI_00001_.png")]}}
        media, errors = handler.collect_output_media(outputs, "job-1")
        assert errors == []
        assert media[0]["kind"] == "image"
        assert media[0]["type"] == "base64"
        assert media[0]["data"] == base64.b64encode(PNG_BYTES).decode()

    def test_temp_files_are_skipped(self, fetch_bytes):
        outputs = {
            "9": {
                "images": [
                    entry("preview.png", file_type="temp"),
                    entry("final.png"),
                ]
            }
        }
        media, errors = handler.collect_output_media(outputs, "job-1")
        assert [m["filename"] for m in media] == ["final.png"]

    def test_missing_filename_is_recorded_as_error(self, fetch_bytes):
        outputs = {"9": {"images": [{"subfolder": "", "type": "output"}]}}
        media, errors = handler.collect_output_media(outputs, "job-1")
        assert media == []
        assert len(errors) == 1

    def test_failed_fetch_is_recorded_as_error(self):
        with patch.object(handler, "get_image_data", return_value=None):
            outputs = {"9": {"images": [entry("gone.png")]}}
            media, errors = handler.collect_output_media(outputs, "job-1")
        assert media == []
        assert "gone.png" in errors[0]

    def test_unrelated_output_keys_are_ignored_not_fatal(self, fetch_bytes):
        outputs = {"3": {"text": ["some string"], "images": [entry("a.png")]}}
        media, errors = handler.collect_output_media(outputs, "job-1")
        assert len(media) == 1
        assert errors == []


class TestMixedWorkflow:
    def test_image_and_video_outputs_are_both_collected(self, fetch_bytes):
        outputs = {
            "9": {"images": [entry("frame_preview.png")]},
            "12": {"gifs": [entry("final_video.mp4")]},
        }
        media, errors = handler.collect_output_media(outputs, "job-1")
        kinds = sorted(m["kind"] for m in media)
        assert kinds == ["image", "video"]


class TestS3Upload:
    def test_videos_upload_to_s3_when_bucket_configured(self, fetch_bytes):
        with patch.dict(os.environ, {"BUCKET_ENDPOINT_URL": "http://bucket"}):
            with patch.object(
                handler.rp_upload,
                "upload_image",
                return_value="http://bucket/out/final_video.mp4",
            ) as mock_upload:
                outputs = {"12": {"gifs": [entry("final_video.mp4")]}}
                media, errors = handler.collect_output_media(outputs, "job-42")
        assert media[0]["type"] == "s3_url"
        assert media[0]["data"] == "http://bucket/out/final_video.mp4"
        assert media[0]["kind"] == "video"
        # uploaded file carried the right extension so the URL is playable
        uploaded_path = mock_upload.call_args.args[1]
        assert uploaded_path.endswith(".mp4")

    def test_s3_failure_is_an_error_not_a_crash(self, fetch_bytes):
        with patch.dict(os.environ, {"BUCKET_ENDPOINT_URL": "http://bucket"}):
            with patch.object(
                handler.rp_upload, "upload_image", side_effect=RuntimeError("boom")
            ):
                outputs = {"12": {"gifs": [entry("final_video.mp4")]}}
                media, errors = handler.collect_output_media(outputs, "job-42")
        assert media == []
        assert "final_video.mp4" in errors[0]


class TestResponseContract:
    """The split into response 'images' and 'videos' lists, without 'kind' leaking."""

    def _assemble(self, output_data, errors):
        """Mirror of handler's final response assembly for collected media."""
        final_result = {}
        images = [
            {k: v for k, v in item.items() if k != "kind"}
            for item in output_data
            if item.get("kind") != "video"
        ]
        videos = [
            {k: v for k, v in item.items() if k != "kind"}
            for item in output_data
            if item.get("kind") == "video"
        ]
        if images:
            final_result["images"] = images
        if videos:
            final_result["videos"] = videos
        return final_result

    def test_video_only_workflow_returns_videos_without_images_key(self, fetch_bytes):
        outputs = {"12": {"gifs": [entry("only_video.mp4")]}}
        media, _ = handler.collect_output_media(outputs, "job-1")
        result = self._assemble(media, [])
        assert "videos" in result and "images" not in result
        assert "kind" not in result["videos"][0]

    def test_image_only_workflow_has_no_videos_key(self, fetch_bytes):
        outputs = {"9": {"images": [entry("a.png")]}}
        media, _ = handler.collect_output_media(outputs, "job-1")
        result = self._assemble(media, [])
        assert "images" in result and "videos" not in result
        assert result["images"][0] == {
            "filename": "a.png",
            "type": "base64",
            "data": base64.b64encode(PNG_BYTES).decode(),
        }
