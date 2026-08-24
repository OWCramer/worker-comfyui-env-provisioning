"""Persona: Priya — wants one custom checkpoint from Civitai, no Docker.

Priya found a checkpoint on Civitai she likes. She deploys the base image
and pastes the model page URL into CHECKPOINT_URLS in the console. She has
never written a Dockerfile and never should have to. The worker must resolve
the URL, figure out the real filename, put the file where ComfyUI looks for
checkpoints, and tell her what filename to reference in her workflow.
"""

import pytest

from provisioning import DownloadError, ResolutionError
from tests.conftest import FakeResponse

DREAMSHAPER_PAGE = "https://civitai.com/models/4384/dreamshaper"


@pytest.fixture
def dreamshaper(worker):
    worker.civitai.add_model(
        4384,
        model_type="Checkpoint",
        versions=[
            {"id": 128713, "filename": "dreamshaper_8.safetensors", "content": b"v8"},
            {"id": 109123, "filename": "dreamshaper_7.safetensors", "content": b"v7"},
        ],
    )
    return worker


class TestCivitaiModelPageUrl:
    """She pastes the *page* URL from her browser — the most natural input."""

    def test_downloads_to_the_checkpoints_directory(self, dreamshaper):
        dreamshaper.environ["CHECKPOINT_URLS"] = DREAMSHAPER_PAGE
        dreamshaper.provision()
        path = dreamshaper.model_path("checkpoints", "dreamshaper_8.safetensors")
        assert path.read_bytes() == b"v8"

    def test_uses_the_newest_version_by_default(self, dreamshaper):
        dreamshaper.environ["CHECKPOINT_URLS"] = DREAMSHAPER_PAGE
        manifest = dreamshaper.provision()
        assert manifest.models[0].filename == "dreamshaper_8.safetensors"

    def test_respects_an_explicit_version_in_the_url(self, dreamshaper):
        dreamshaper.environ["CHECKPOINT_URLS"] = (
            f"{DREAMSHAPER_PAGE}?modelVersionId=109123"
        )
        dreamshaper.provision()
        path = dreamshaper.model_path("checkpoints", "dreamshaper_7.safetensors")
        assert path.read_bytes() == b"v7"

    def test_unknown_version_id_fails_with_available_versions(self, dreamshaper):
        dreamshaper.environ["CHECKPOINT_URLS"] = (
            f"{DREAMSHAPER_PAGE}?modelVersionId=999999"
        )
        with pytest.raises(ResolutionError) as excinfo:
            dreamshaper.provision()
        assert "128713" in str(excinfo.value)

    def test_manifest_tells_her_the_workflow_filename(self, dreamshaper):
        dreamshaper.environ["CHECKPOINT_URLS"] = DREAMSHAPER_PAGE
        dreamshaper.provision()
        manifest = dreamshaper.manifest_json()
        assert manifest["models"][0]["filename"] == "dreamshaper_8.safetensors"
        assert manifest["models"][0]["directory"] == "checkpoints"


class TestCivitaiDownloadUrl:
    """Power-user variant: she copied the download button link instead."""

    def test_resolves_filename_via_the_version_api(self, dreamshaper):
        dreamshaper.environ["CHECKPOINT_URLS"] = (
            "https://civitai.com/api/download/models/128713"
        )
        dreamshaper.provision()
        assert dreamshaper.model_path(
            "checkpoints", "dreamshaper_8.safetensors"
        ).exists()

    def test_filename_override_wins(self, dreamshaper):
        dreamshaper.environ["CHECKPOINT_URLS"] = (
            "https://civitai.com/api/download/models/128713::my_favorite.safetensors"
        )
        dreamshaper.provision()
        assert dreamshaper.model_path("checkpoints", "my_favorite.safetensors").exists()
        assert not dreamshaper.model_path(
            "checkpoints", "dreamshaper_8.safetensors"
        ).exists()


class TestCivitaiAuthentication:
    @pytest.fixture
    def gated_model(self, worker):
        worker.civitai.add_model(
            777,
            versions=[{"id": 7770, "filename": "gated.safetensors"}],
            requires_token=True,
        )
        worker.environ["CHECKPOINT_URLS"] = "https://civitai.com/models/777/gated"
        return worker

    def test_token_is_sent_to_the_api_and_download(self, gated_model):
        gated_model.environ["CIVITAI_TOKEN"] = "civ-secret"
        gated_model.provision()
        api_url, api_headers = gated_model.session.requests[0]
        assert api_headers.get("Authorization") == "Bearer civ-secret"
        download_urls = [
            url for url in gated_model.session.requested_urls() if "download" in url
        ]
        assert any("token=civ-secret" in url for url in download_urls)

    def test_missing_token_error_names_the_env_var(self, gated_model):
        with pytest.raises(ResolutionError) as excinfo:
            gated_model.provision()
        assert "CIVITAI_TOKEN" in str(excinfo.value)

    def test_civitai_api_token_alias_is_accepted(self, gated_model):
        gated_model.environ["CIVITAI_API_TOKEN"] = "civ-secret"
        gated_model.provision()
        assert gated_model.model_path("checkpoints", "gated.safetensors").exists()


class TestHuggingFaceUrl:
    HF_URL = "https://huggingface.co/stabilityai/sdxl-vae/resolve/main/sdxl_vae.safetensors"

    def test_filename_comes_from_the_url_path(self, worker):
        worker.session.register(self.HF_URL, FakeResponse(content=b"vae-bytes"))
        worker.environ["VAE_URLS"] = self.HF_URL
        worker.provision()
        assert worker.model_path("vae", "sdxl_vae.safetensors").read_bytes() == b"vae-bytes"

    def test_hf_token_is_sent_as_bearer_header(self, worker):
        worker.session.register(self.HF_URL, FakeResponse(content=b"vae-bytes"))
        worker.environ["VAE_URLS"] = self.HF_URL
        worker.environ["HF_TOKEN"] = "hf-secret"
        worker.provision()
        _, headers = worker.session.requests[-1]
        assert headers.get("Authorization") == "Bearer hf-secret"

    def test_huggingface_access_token_alias_is_accepted(self, worker):
        worker.session.register(self.HF_URL, FakeResponse(content=b"vae-bytes"))
        worker.environ["VAE_URLS"] = self.HF_URL
        worker.environ["HUGGINGFACE_ACCESS_TOKEN"] = "hf-secret"
        worker.provision()
        _, headers = worker.session.requests[-1]
        assert headers.get("Authorization") == "Bearer hf-secret"

    def test_gated_model_403_error_names_hf_token(self, worker):
        worker.session.register(self.HF_URL, FakeResponse(status_code=403))
        worker.environ["VAE_URLS"] = self.HF_URL
        with pytest.raises(DownloadError) as excinfo:
            worker.provision()
        assert "HF_TOKEN" in str(excinfo.value)


class TestDirectUrl:
    def test_any_url_with_a_filename_in_the_path_works(self, worker):
        url = "https://example.com/models/custom_model.safetensors"
        worker.session.register(url, FakeResponse(content=b"direct"))
        worker.environ["CHECKPOINT_URLS"] = url
        worker.provision()
        assert worker.model_path("checkpoints", "custom_model.safetensors").exists()

    def test_url_without_a_filename_explains_the_override_syntax(self, worker):
        worker.environ["CHECKPOINT_URLS"] = "https://example.com/download?id=42"
        with pytest.raises(ResolutionError) as excinfo:
            worker.provision()
        assert "::" in str(excinfo.value)


class TestPlacement:
    def test_without_a_volume_models_land_in_the_image_tree(
        self, worker_no_volume
    ):
        worker_no_volume.civitai.add_model(
            4384, versions=[{"id": 128713, "filename": "dreamshaper_8.safetensors"}]
        )
        worker_no_volume.environ["CHECKPOINT_URLS"] = DREAMSHAPER_PAGE
        worker_no_volume.provision()
        assert (
            worker_no_volume.comfy_home
            / "models/checkpoints/dreamshaper_8.safetensors"
        ).exists()

    def test_with_a_volume_models_land_on_the_volume(self, dreamshaper):
        dreamshaper.environ["CHECKPOINT_URLS"] = DREAMSHAPER_PAGE
        dreamshaper.provision()
        assert (
            dreamshaper.volume_path / "models/checkpoints/dreamshaper_8.safetensors"
        ).exists()

    def test_type_mismatch_warns_but_still_downloads(self, worker):
        worker.civitai.add_model(
            555,
            model_type="LORA",
            versions=[{"id": 5550, "filename": "detail_tweaker.safetensors"}],
        )
        worker.environ["CHECKPOINT_URLS"] = "https://civitai.com/models/555/oops"
        manifest = worker.provision()
        assert worker.model_path("checkpoints", "detail_tweaker.safetensors").exists()
        assert any("loras" in w for w in manifest.warnings)
