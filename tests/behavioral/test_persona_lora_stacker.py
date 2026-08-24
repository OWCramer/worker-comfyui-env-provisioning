"""Persona: Marco — a checkpoint plus a stack of LoRAs, swapped weekly.

Marco runs a product-photography endpoint: one base checkpoint, several
LoRAs, a VAE. He tweaks the model list every few days by editing env vars —
no rebuilds. He runs a network volume so his fleet downloads each file once,
and he scales to many workers, so concurrent cold starts are his normal.
"""

import os
import time

import pytest

from provisioning import DownloadError
from tests.conftest import FakeResponse

CHECKPOINT_URL = "https://example.com/files/photo_base.safetensors"
LORA_URLS = [
    "https://example.com/files/soft_light.safetensors",
    "https://example.com/files/product_detail.safetensors",
    "https://example.com/files/white_background.safetensors",
]
VAE_URL = "https://example.com/files/color_fix.safetensors"


@pytest.fixture
def marco(worker):
    for url in [CHECKPOINT_URL, VAE_URL, *LORA_URLS]:
        worker.session.register(url, FakeResponse(content=b"bytes:" + url.encode()))
    worker.environ["CHECKPOINT_URLS"] = CHECKPOINT_URL
    worker.environ["LORA_URLS"] = ", ".join(LORA_URLS)
    worker.environ["VAE_URLS"] = VAE_URL
    return worker


class TestMultiModelSetup:
    def test_every_model_lands_in_its_directory(self, marco):
        manifest = marco.provision()
        assert marco.model_path("checkpoints", "photo_base.safetensors").exists()
        assert marco.model_path("vae", "color_fix.safetensors").exists()
        for url in LORA_URLS:
            assert marco.model_path("loras", url.rsplit("/", 1)[-1]).exists()
        assert len(manifest.models) == 5

    def test_newline_separated_entries_work_too(self, marco):
        marco.environ["LORA_URLS"] = "\n".join(LORA_URLS)
        manifest = marco.provision()
        assert len(manifest.models) == 5

    def test_trailing_commas_and_blank_lines_are_ignored(self, marco):
        marco.environ["LORA_URLS"] = f"{LORA_URLS[0]},\n\n  ,{LORA_URLS[1]},"
        manifest = marco.provision()
        lora_results = [m for m in manifest.models if m.directory == "loras"]
        assert len(lora_results) == 2

    def test_a_url_listed_twice_downloads_once(self, marco):
        marco.environ["LORA_URLS"] = f"{LORA_URLS[0]}, {LORA_URLS[0]}"
        marco.provision()
        assert marco.session.download_count(LORA_URLS[0]) == 1

    def test_same_filename_in_different_directories_is_fine(self, worker):
        url_a = "https://example.com/a/model.safetensors"
        url_b = "https://example.com/b/model.safetensors"
        worker.session.register(url_a, FakeResponse(content=b"AAA"))
        worker.session.register(url_b, FakeResponse(content=b"BBB"))
        worker.environ["CHECKPOINT_URLS"] = url_a
        worker.environ["LORA_URLS"] = url_b
        worker.provision()
        assert worker.model_path("checkpoints", "model.safetensors").read_bytes() == b"AAA"
        assert worker.model_path("loras", "model.safetensors").read_bytes() == b"BBB"


class TestFleetEfficiency:
    """One download per file per fleet — that's what the volume is for."""

    def test_second_cold_start_downloads_nothing(self, marco):
        marco.provision()
        first_request_count = len(marco.session.requests)
        manifest = marco.provision()  # a second worker boots
        assert len(marco.session.requests) == first_request_count
        assert all(m.status == "skipped" for m in manifest.models)

    def test_adding_one_lora_downloads_only_that_lora(self, marco):
        marco.provision()
        new_url = "https://example.com/files/film_grain.safetensors"
        marco.session.register(new_url, FakeResponse(content=b"new"))
        marco.environ["LORA_URLS"] += f", {new_url}"
        manifest = marco.provision()
        downloaded = [m for m in manifest.models if m.status == "downloaded"]
        assert [m.filename for m in downloaded] == ["film_grain.safetensors"]

    def test_a_leftover_part_file_is_not_treated_as_complete(self, marco):
        # A worker crashed mid-download on a previous boot.
        part = marco.model_path("checkpoints", "photo_base.safetensors.part")
        part.parent.mkdir(parents=True, exist_ok=True)
        part.write_bytes(b"trunc")
        marco.provision()
        final = marco.model_path("checkpoints", "photo_base.safetensors")
        assert final.read_bytes() == b"bytes:" + CHECKPOINT_URL.encode()
        assert not part.exists()


class TestConcurrentColdStarts:
    def test_waits_for_the_other_worker_then_skips(self, marco):
        final = marco.model_path("checkpoints", "photo_base.safetensors")
        final.parent.mkdir(parents=True, exist_ok=True)
        lock = final.with_name(final.name + ".lock")
        lock.write_text("other-worker")

        # While "waiting", the other worker finishes the file.
        def sleep_and_finish(seconds):
            final.write_bytes(b"finished-by-other-worker")

        manifest = marco.provision(sleep=sleep_and_finish)
        checkpoint = next(m for m in manifest.models if m.directory == "checkpoints")
        assert checkpoint.status == "skipped"
        assert final.read_bytes() == b"finished-by-other-worker"
        assert marco.session.download_count(CHECKPOINT_URL) == 0

    def test_a_stale_lock_from_a_crashed_worker_is_broken(self, marco):
        final = marco.model_path("checkpoints", "photo_base.safetensors")
        final.parent.mkdir(parents=True, exist_ok=True)
        lock = final.with_name(final.name + ".lock")
        lock.write_text("crashed-worker")
        stale = time.time() - 7200
        os.utime(lock, (stale, stale))

        manifest = marco.provision()
        checkpoint = next(m for m in manifest.models if m.directory == "checkpoints")
        assert checkpoint.status == "downloaded"
        assert final.exists()

    def test_gives_up_with_guidance_if_the_lock_never_clears(self, marco):
        final = marco.model_path("checkpoints", "photo_base.safetensors")
        final.parent.mkdir(parents=True, exist_ok=True)
        lock = final.with_name(final.name + ".lock")
        lock.write_text("wedged-worker")

        def touch_lock(seconds):
            os.utime(lock)  # keeps the lock fresh: holder alive but slow/wedged

        with pytest.raises(DownloadError) as excinfo:
            marco.provision(sleep=touch_lock, lock_wait_seconds=3.0)
        assert "lock" in str(excinfo.value).lower()


class TestFlakySources:
    def test_transient_server_error_is_retried(self, marco):
        marco.session.register(
            CHECKPOINT_URL,
            [FakeResponse(status_code=502), FakeResponse(content=b"recovered")],
        )
        marco.provision()
        assert (
            marco.model_path("checkpoints", "photo_base.safetensors").read_bytes()
            == b"recovered"
        )

    def test_persistent_server_error_fails_with_the_url(self, marco):
        marco.session.register(CHECKPOINT_URL, FakeResponse(status_code=500))
        with pytest.raises(DownloadError) as excinfo:
            marco.provision()
        assert CHECKPOINT_URL in str(excinfo.value)

    def test_connection_errors_are_retried(self, marco):
        state = {"calls": 0}

        def flaky(url, headers):
            state["calls"] += 1
            if state["calls"] == 1:
                raise ConnectionError("reset by peer")
            return FakeResponse(content=b"recovered")

        marco.session.register(CHECKPOINT_URL, flaky)
        marco.provision()
        assert (
            marco.model_path("checkpoints", "photo_base.safetensors").read_bytes()
            == b"recovered"
        )
