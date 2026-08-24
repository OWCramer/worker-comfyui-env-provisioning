"""Persona: Dana — deploys a packaged image and configures nothing.

Dana picks the flux1-dev image from the Hub, sets zero provisioning env
vars, and expects the worker to behave exactly as it does today. This file
is the regression guard: provisioning must be invisible when unused.
"""

import pytest


class TestNoConfiguration:
    def test_no_env_vars_is_a_noop(self, worker):
        manifest = worker.provision()
        assert manifest.is_noop

    def test_no_env_vars_makes_no_http_requests(self, worker):
        worker.provision()
        assert worker.session.requests == []

    def test_no_env_vars_runs_no_subprocesses(self, worker):
        worker.provision()
        assert worker.node_runner.calls == []

    def test_no_env_vars_writes_no_manifest_or_env_file(self, worker):
        worker.provision()
        assert not worker.manifest_path.exists()
        assert not worker.env_file.exists()

    def test_no_env_vars_leaves_volume_untouched(self, worker):
        worker.provision()
        assert list(worker.volume_path.iterdir()) == []

    def test_works_without_a_network_volume(self, worker_no_volume):
        manifest = worker_no_volume.provision()
        assert manifest.is_noop

    def test_empty_string_env_vars_are_treated_as_unset(self, worker):
        worker.environ.update(
            {"CHECKPOINT_URLS": "", "LORA_URLS": "   ", "CUSTOM_NODES": "\n , \n"}
        )
        manifest = worker.provision()
        assert manifest.is_noop
        assert worker.session.requests == []

    def test_baked_models_are_left_alone(self, worker_no_volume):
        baked = worker_no_volume.bake_model("checkpoints", "sd_xl_base_1.0.safetensors")
        worker_no_volume.provision()
        assert baked.read_bytes() == b"baked"


class TestKillSwitch:
    """RUNTIME_PROVISIONING defaults on; setting it false disables everything."""

    @pytest.fixture
    def configured_worker(self, worker):
        worker.civitai.add_model(
            101, versions=[{"id": 9001, "filename": "some_model.safetensors"}]
        )
        worker.environ["CHECKPOINT_URLS"] = "https://civitai.com/models/101/some-model"
        worker.environ["CUSTOM_NODES"] = "comfyui-kjnodes@1.0.0"
        return worker

    @pytest.mark.parametrize("value", ["false", "FALSE", "0", "no", "off"])
    def test_kill_switch_disables_provisioning(self, configured_worker, value):
        configured_worker.environ["RUNTIME_PROVISIONING"] = value
        manifest = configured_worker.provision()
        assert manifest.is_noop
        assert not manifest.enabled
        assert configured_worker.session.requests == []
        assert configured_worker.node_runner.calls == []

    @pytest.mark.parametrize("value", ["true", "TRUE", "1", "yes", "on", ""])
    def test_provisioning_is_enabled_by_default_and_for_truthy_values(
        self, configured_worker, value
    ):
        if value:
            configured_worker.environ["RUNTIME_PROVISIONING"] = value
        manifest = configured_worker.provision()
        assert not manifest.is_noop
        assert manifest.enabled
