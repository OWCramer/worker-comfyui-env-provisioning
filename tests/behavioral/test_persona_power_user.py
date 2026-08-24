"""Persona: Ken — a production Flux pipeline with everything at once.

Ken ships a robust workflow: a diffusion model, text encoders, a VAE,
ControlNets, LoRAs, upscalers, and several custom nodes. His config mixes
Civitai, Hugging Face, and direct URLs. He deploys to a large endpoint and
expects provisioning to be deterministic, ordered (nodes before models),
fully inventoried, and to fail fast and loud when one piece breaks.
"""

import pytest

from provisioning import DownloadError
from provisioning.__main__ import main as provisioning_main
from tests.conftest import FakeResponse

HF = "https://huggingface.co"
MODELS = {
    "DIFFUSION_MODEL_URLS": [
        f"{HF}/black-forest-labs/FLUX.1-dev/resolve/main/flux1-dev.safetensors"
    ],
    "TEXT_ENCODER_URLS": [
        f"{HF}/comfyanonymous/flux_text_encoders/resolve/main/clip_l.safetensors",
        f"{HF}/comfyanonymous/flux_text_encoders/resolve/main/t5xxl_fp8_e4m3fn.safetensors",
    ],
    "VAE_URLS": [f"{HF}/black-forest-labs/FLUX.1-dev/resolve/main/ae.safetensors"],
    "CONTROLNET_URLS": [
        "https://example.com/cn/flux-canny.safetensors",
        "https://example.com/cn/flux-depth.safetensors",
    ],
    "UPSCALE_MODEL_URLS": ["https://example.com/up/4x_ultrasharp.pth"],
    "LORA_URLS": ["https://civitai.com/models/652699/amateur-photography"],
}
EXPECTED_DIRS = {
    "DIFFUSION_MODEL_URLS": "diffusion_models",
    "TEXT_ENCODER_URLS": "text_encoders",
    "VAE_URLS": "vae",
    "CONTROLNET_URLS": "controlnet",
    "UPSCALE_MODEL_URLS": "upscale_models",
    "LORA_URLS": "loras",
}


@pytest.fixture
def ken(worker):
    for urls in MODELS.values():
        for url in urls:
            if "civitai" not in url:
                worker.session.register(url, FakeResponse(content=b"bytes"))
    worker.civitai.add_model(
        652699,
        model_type="LORA",
        versions=[{"id": 993999, "filename": "amateur_photography_v6.safetensors"}],
    )
    worker.environ.update({var: ", ".join(urls) for var, urls in MODELS.items()})
    worker.environ["CUSTOM_NODES"] = (
        "comfyui-kjnodes@1.1.2, comfyui-easy-use@1.3.0, rgthree-comfy@1.0.0"
    )
    return worker


class TestKitchenSink:
    def test_every_model_lands_in_the_right_directory(self, ken):
        manifest = ken.provision()
        assert len(manifest.models) == 8
        for var, urls in MODELS.items():
            directory = EXPECTED_DIRS[var]
            results = [m for m in manifest.models if m.directory == directory]
            assert len(results) == len(urls), f"missing models for {var}"
            for m in results:
                assert ken.model_path(directory, m.filename).exists()

    def test_all_custom_nodes_are_installed(self, ken):
        ken.provision()
        assert len(ken.node_runner.install_calls()) == 3

    def test_nodes_are_installed_before_models_download(self, ken):
        events = []
        original_get = ken.session.get
        original_runner = ken.node_runner.__call__

        def tracking_get(*args, **kwargs):
            events.append("model-traffic")
            return original_get(*args, **kwargs)

        class TrackingRunner:
            def __init__(self, inner):
                self.inner = inner

            def __getattr__(self, item):
                return getattr(self.inner, item)

            def __call__(self, *args, **kwargs):
                events.append("node-install")
                return self.inner(*args, **kwargs)

        ken.session.get = tracking_get
        ken.provision(node_runner=TrackingRunner(ken.node_runner))
        first_model = events.index("model-traffic")
        assert all(e == "node-install" for e in events[:first_model])
        assert "node-install" not in events[first_model:]

    def test_manifest_json_is_a_complete_inventory(self, ken):
        ken.provision()
        manifest = ken.manifest_json()
        assert len(manifest["models"]) == 8
        assert len(manifest["nodes"]) == 3
        assert manifest["enabled"] is True
        filenames = {m["filename"] for m in manifest["models"]}
        assert "amateur_photography_v6.safetensors" in filenames

    def test_rerunning_is_fully_deterministic_and_idle(self, ken):
        ken.provision()
        model_requests = len(ken.session.requests)
        ken.node_runner.calls.clear()

        manifest = ken.provision()
        assert all(m.status == "skipped" for m in manifest.models)
        assert ken.node_runner.install_calls() == []
        # Only Civitai metadata may be re-fetched; no file downloads happen.
        new_urls = ken.session.requested_urls()[model_requests:]
        assert not [u for u in new_urls if "example.com" in u or "huggingface" in u]


class TestScale:
    def test_thirty_models_in_one_variable(self, worker):
        urls = []
        for i in range(30):
            url = f"https://example.com/loras/style_{i:02d}.safetensors"
            worker.session.register(url, FakeResponse(content=b"x"))
            urls.append(url)
        worker.environ["LORA_URLS"] = ",".join(urls)
        manifest = worker.provision()
        assert len(manifest.models) == 30
        assert all(m.status == "downloaded" for m in manifest.models)

    def test_query_strings_are_preserved_on_direct_downloads(self, worker):
        url = "https://example.com/get/model.safetensors?sig=abc123&expires=999"
        worker.session.register(url, FakeResponse(content=b"signed"))
        worker.environ["CHECKPOINT_URLS"] = url
        worker.provision()
        requested = worker.session.requested_urls()[0]
        assert "sig=abc123" in requested and "expires=999" in requested

    def test_filenames_with_spaces_are_allowed(self, worker):
        url = "https://example.com/m/base.safetensors"
        worker.session.register(url, FakeResponse(content=b"x"))
        worker.environ["CHECKPOINT_URLS"] = f"{url}::my favorite model.safetensors"
        worker.provision()
        assert worker.model_path("checkpoints", "my favorite model.safetensors").exists()


class TestFailFast:
    def test_one_broken_model_fails_the_whole_provision(self, ken):
        ken.session.register(
            "https://example.com/cn/flux-depth.safetensors",
            FakeResponse(status_code=404),
        )
        with pytest.raises(DownloadError) as excinfo:
            ken.provision()
        assert "flux-depth" in str(excinfo.value)
        assert "CONTROLNET_URLS" in str(excinfo.value)

    def test_cli_entrypoint_exits_nonzero_on_failure(self, ken, monkeypatch, capsys):
        ken.session.register(
            "https://example.com/cn/flux-depth.safetensors",
            FakeResponse(status_code=404),
        )

        def failing_provision(*args, **kwargs):
            return ken.provision()

        monkeypatch.setattr("provisioning.__main__.provision", failing_provision)
        assert provisioning_main() == 1
        assert "PROVISIONING FAILED" in capsys.readouterr().err

    def test_cli_entrypoint_exits_zero_on_success(self, ken, monkeypatch):
        monkeypatch.setattr(
            "provisioning.__main__.provision", lambda *a, **k: ken.provision()
        )
        assert provisioning_main() == 0
