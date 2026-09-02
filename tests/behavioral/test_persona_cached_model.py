"""Persona: Sam — deploys a model Runpod has already cached on the host.

Sam picked his model in the console's Model field, so Runpod cached the
repository onto the machines his workers land on and mounted it under the
network volume. He should not then wait for the worker to download the same
file again: the whole point of caching is that his first request starts
immediately instead of pulling tens of gigabytes inside the job.

The worker must find the cached file, put it where ComfyUI looks under the
filename the workflow references, and say it did so — while still downloading
normally whenever the cache has nothing to offer.
"""

import os

from tests.conftest import FakeResponse

FLUX_URL = (
    "https://huggingface.co/Comfy-Org/flux1-dev/resolve/main/flux1-dev-fp8.safetensors"
)


class TestCacheHit:
    """The file Runpod cached is the file the worker uses."""

    def test_uses_the_cached_file_instead_of_downloading(self, worker):
        worker.cache_model("Comfy-Org/flux1-dev", "flux1-dev-fp8.safetensors", b"weights")
        worker.environ["CHECKPOINT_URLS"] = FLUX_URL

        worker.provision()

        path = worker.model_path("checkpoints", "flux1-dev-fp8.safetensors")
        assert path.read_bytes() == b"weights"
        assert worker.session.requests == []

    def test_honours_the_filename_the_workflow_references(self, worker):
        worker.cache_model("Comfy-Org/flux1-dev", "flux1-dev-fp8.safetensors", b"weights")
        worker.environ["CHECKPOINT_URLS"] = f"{FLUX_URL}::model.safetensors"

        worker.provision()

        assert worker.model_path("checkpoints", "model.safetensors").read_bytes() == b"weights"

    def test_links_rather_than_copying_the_file(self, worker):
        """A cached checkpoint can be tens of gigabytes; copying it defeats the point."""
        worker.cache_model("Comfy-Org/flux1-dev", "flux1-dev-fp8.safetensors", b"weights")
        worker.environ["CHECKPOINT_URLS"] = FLUX_URL

        worker.provision()

        assert os.path.islink(worker.model_path("checkpoints", "flux1-dev-fp8.safetensors"))

    def test_reports_the_file_as_cached(self, worker):
        worker.cache_model("Comfy-Org/flux1-dev", "flux1-dev-fp8.safetensors", b"weights")
        worker.environ["CHECKPOINT_URLS"] = FLUX_URL

        manifest = worker.provision()

        assert [m.status for m in manifest.models] == ["cached"]

    def test_finds_a_file_nested_inside_the_repository(self, worker):
        worker.cache_model(
            "Comfy-Org/Wan_2.2_ComfyUI_Repackaged",
            "split_files/vae/wan2.2_vae.safetensors",
            b"vae",
        )
        worker.environ["VAE_URLS"] = (
            "https://huggingface.co/Comfy-Org/Wan_2.2_ComfyUI_Repackaged"
            "/resolve/main/split_files/vae/wan2.2_vae.safetensors"
        )

        worker.provision()

        assert worker.model_path("vae", "wan2.2_vae.safetensors").read_bytes() == b"vae"

    def test_reads_the_snapshot_the_main_ref_names(self, worker):
        worker.cache_model(
            "Comfy-Org/flux1-dev", "flux1-dev-fp8.safetensors", b"old", revision="aaa"
        )
        worker.cache_model(
            "Comfy-Org/flux1-dev", "flux1-dev-fp8.safetensors", b"current", revision="zzz"
        )
        worker.environ["CHECKPOINT_URLS"] = FLUX_URL

        worker.provision()

        # cache_model rewrites refs/main, so "zzz" is current despite sorting last.
        path = worker.model_path("checkpoints", "flux1-dev-fp8.safetensors")
        assert path.read_bytes() == b"current"


class TestCacheMiss:
    """Anything the cache cannot serve still downloads exactly as before."""

    def test_downloads_when_the_repository_is_not_cached(self, worker):
        worker.session.register(FLUX_URL, FakeResponse(content=b"fetched"))
        worker.environ["CHECKPOINT_URLS"] = FLUX_URL

        manifest = worker.provision()

        path = worker.model_path("checkpoints", "flux1-dev-fp8.safetensors")
        assert path.read_bytes() == b"fetched"
        assert [m.status for m in manifest.models] == ["downloaded"]

    def test_downloads_when_the_repository_is_cached_without_that_file(self, worker):
        worker.cache_model("Comfy-Org/flux1-dev", "flux1-dev.safetensors", b"other")
        worker.session.register(FLUX_URL, FakeResponse(content=b"fetched"))
        worker.environ["CHECKPOINT_URLS"] = FLUX_URL

        worker.provision()

        path = worker.model_path("checkpoints", "flux1-dev-fp8.safetensors")
        assert path.read_bytes() == b"fetched"

    def test_leaves_civitai_downloads_alone(self, worker):
        """Only Hugging Face repositories are cached, so Civitai must be untouched."""
        worker.civitai.add_model(
            4384,
            model_type="Checkpoint",
            versions=[{"id": 1, "filename": "dreamshaper_8.safetensors", "content": b"v8"}],
        )
        worker.environ["CHECKPOINT_URLS"] = "https://civitai.com/models/4384/dreamshaper"

        manifest = worker.provision()

        assert [m.status for m in manifest.models] == ["downloaded"]

    def test_a_file_already_in_place_still_wins(self, worker):
        """An existing download is authoritative; the cache must not overwrite it."""
        existing = worker.model_path("checkpoints", "flux1-dev-fp8.safetensors")
        existing.parent.mkdir(parents=True, exist_ok=True)
        existing.write_bytes(b"already here")
        worker.cache_model("Comfy-Org/flux1-dev", "flux1-dev-fp8.safetensors", b"weights")
        worker.environ["CHECKPOINT_URLS"] = FLUX_URL

        manifest = worker.provision()

        assert existing.read_bytes() == b"already here"
        assert [m.status for m in manifest.models] == ["skipped"]

    def test_downloads_normally_without_a_volume(self, worker_no_volume):
        worker_no_volume.session.register(FLUX_URL, FakeResponse(content=b"fetched"))
        worker_no_volume.environ["CHECKPOINT_URLS"] = FLUX_URL

        manifest = worker_no_volume.provision()

        assert [m.status for m in manifest.models] == ["downloaded"]
