"""Persona: Sam — deploys a model through Runpod's shared Hugging Face cache.

Sam picked his model in the console's Model field, so Runpod cached that
repository onto the machines his workers land on and mounted it under the
network volume. He should not then wait for the worker to fetch the same file
again: the point of the cache is that his first request starts immediately.

The worker reaches Hugging Face through that cache the way worker-vllm does, so
two things have to hold. A file already cached costs nothing. A file that is not
is fetched *into* the cache, so his second worker — and his next endpoint on the
same volume — gets it free. That second part matters because the platform
pre-caches one repository per endpoint, while a video model needs three.
"""

import os

REPO = "Comfy-Org/flux1-dev"
FILE = "flux1-dev-fp8.safetensors"
FLUX_URL = f"https://huggingface.co/{REPO}/resolve/main/{FILE}"


class TestAlreadyCached:
    """The repository Runpod cached is used as it stands."""

    def test_uses_the_cached_file_without_downloading_it(self, worker):
        worker.hub.preload(REPO, FILE, b"weights")
        worker.environ["CHECKPOINT_URLS"] = FLUX_URL

        worker.provision()

        assert worker.model_path("checkpoints", FILE).read_bytes() == b"weights"
        assert worker.session.requested_urls() == []

    def test_honours_the_filename_the_workflow_references(self, worker):
        worker.hub.preload(REPO, FILE, b"weights")
        worker.environ["CHECKPOINT_URLS"] = f"{FLUX_URL}::model.safetensors"

        worker.provision()

        assert worker.model_path("checkpoints", "model.safetensors").read_bytes() == b"weights"

    def test_links_rather_than_copying_the_file(self, worker):
        """A checkpoint can be tens of gigabytes; copying it defeats the point."""
        worker.hub.preload(REPO, FILE, b"weights")
        worker.environ["CHECKPOINT_URLS"] = FLUX_URL

        worker.provision()

        assert os.path.islink(worker.model_path("checkpoints", FILE))

    def test_reports_the_file_as_cached(self, worker):
        worker.hub.preload(REPO, FILE, b"weights")
        worker.environ["CHECKPOINT_URLS"] = FLUX_URL

        manifest = worker.provision()

        assert [m.status for m in manifest.models] == ["cached"]

    def test_asks_for_the_revision_the_link_names(self, worker):
        worker.hub.preload(REPO, FILE, b"weights")
        worker.environ["CHECKPOINT_URLS"] = (
            f"https://huggingface.co/{REPO}/resolve/refs%2Fpr%2F1/{FILE}"
        )

        worker.provision()

        assert worker.hub.calls == [(REPO, FILE, "refs/pr/1")]

    def test_finds_a_file_nested_inside_the_repository(self, worker):
        nested = "split_files/vae/wan2.2_vae.safetensors"
        worker.hub.preload("Comfy-Org/Wan_2.2_ComfyUI_Repackaged", nested, b"vae")
        worker.environ["VAE_URLS"] = (
            f"https://huggingface.co/Comfy-Org/Wan_2.2_ComfyUI_Repackaged/resolve/main/{nested}"
        )

        worker.provision()

        assert worker.model_path("vae", "wan2.2_vae.safetensors").read_bytes() == b"vae"


class TestNotYetCached:
    """Anything the cache lacks is fetched into it, not around it."""

    def test_fetches_through_the_cache_rather_than_over_http(self, worker):
        worker.hub.publish(REPO, FILE, b"weights")
        worker.environ["CHECKPOINT_URLS"] = FLUX_URL

        manifest = worker.provision()

        assert worker.model_path("checkpoints", FILE).read_bytes() == b"weights"
        assert worker.session.requested_urls() == []
        assert [m.status for m in manifest.models] == ["cached"]

    def test_leaves_the_file_cached_for_the_next_worker(self, worker):
        worker.hub.publish(REPO, FILE, b"weights")
        worker.environ["CHECKPOINT_URLS"] = FLUX_URL

        worker.provision()

        assert (REPO, FILE) in worker.hub.cached

    def test_caches_every_file_a_video_model_needs(self, worker):
        """The platform pre-caches one repository; the rest arrive this way."""
        repackaged = "Comfy-Org/Wan_2.2_ComfyUI_Repackaged"
        worker.hub.preload(repackaged, "split_files/diffusion_models/wan.safetensors", b"unet")
        worker.hub.publish(repackaged, "split_files/text_encoders/umt5.safetensors", b"clip")
        worker.hub.publish(repackaged, "split_files/vae/wan2.2_vae.safetensors", b"vae")
        base = f"https://huggingface.co/{repackaged}/resolve/main/split_files"
        worker.environ["DIFFUSION_MODEL_URLS"] = f"{base}/diffusion_models/wan.safetensors"
        worker.environ["TEXT_ENCODER_URLS"] = f"{base}/text_encoders/umt5.safetensors"
        worker.environ["VAE_URLS"] = f"{base}/vae/wan2.2_vae.safetensors"

        manifest = worker.provision()

        assert [m.status for m in manifest.models] == ["cached", "cached", "cached"]
        assert worker.session.requested_urls() == []


class TestFallback:
    """The cache is an optimisation, never a new way to fail."""

    def test_downloads_over_http_when_the_hub_cannot_serve_the_file(self, worker):
        from tests.conftest import FakeResponse

        worker.session.register(FLUX_URL, FakeResponse(content=b"fetched"))
        worker.environ["CHECKPOINT_URLS"] = FLUX_URL

        manifest = worker.provision()

        assert worker.model_path("checkpoints", FILE).read_bytes() == b"fetched"
        assert [m.status for m in manifest.models] == ["downloaded"]

    def test_leaves_civitai_alone(self, worker):
        """Only Hugging Face is cached, so Civitai must be untouched by this."""
        worker.civitai.add_model(
            4384,
            model_type="Checkpoint",
            versions=[{"id": 1, "filename": "dreamshaper_8.safetensors", "content": b"v8"}],
        )
        worker.environ["CHECKPOINT_URLS"] = "https://civitai.com/models/4384/dreamshaper"

        manifest = worker.provision()

        assert [m.status for m in manifest.models] == ["downloaded"]
        assert worker.hub.calls == []

    def test_a_file_already_in_place_still_wins(self, worker):
        existing = worker.model_path("checkpoints", FILE)
        existing.parent.mkdir(parents=True, exist_ok=True)
        existing.write_bytes(b"already here")
        worker.hub.preload(REPO, FILE, b"weights")
        worker.environ["CHECKPOINT_URLS"] = FLUX_URL

        manifest = worker.provision()

        assert existing.read_bytes() == b"already here"
        assert [m.status for m in manifest.models] == ["skipped"]

    def test_downloads_normally_without_a_volume_to_cache_into(self, worker_no_volume):
        from tests.conftest import FakeResponse

        worker_no_volume.session.register(FLUX_URL, FakeResponse(content=b"fetched"))
        worker_no_volume.environ["CHECKPOINT_URLS"] = FLUX_URL

        manifest = worker_no_volume.provision()

        assert [m.status for m in manifest.models] == ["downloaded"]
