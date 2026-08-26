"""Persona: Sofia — needs a handful of registry custom nodes, no Docker.

Sofia's workflow uses KJNodes and IC-Light. Today she'd have to write a
Dockerfile; instead she sets CUSTOM_NODES on her endpoint. She pins versions
because she deploys to production, runs a network volume, and expects the
second worker to cold-start without reinstalling anything.
"""

import json
import shutil

import pytest

from provisioning import NodeInstallError
from provisioning.nodes import CACHE_DIR_NAME


@pytest.fixture
def sofia(worker):
    worker.environ["CUSTOM_NODES"] = "comfyui-kjnodes@1.1.2, comfyui-ic-light@1.0.5"
    return worker


class TestBasicInstall:
    def test_each_node_is_installed_with_its_pin(self, sofia):
        sofia.provision()
        assert sofia.node_runner.install_calls() == [
            "comfyui-kjnodes@1.1.2",
            "comfyui-ic-light@1.0.5",
        ]

    def test_node_directories_exist_after_provisioning(self, sofia):
        sofia.provision()
        assert (sofia.comfy_home / "custom_nodes/ComfyuiKjnodes").is_dir()
        assert (sofia.comfy_home / "custom_nodes/ComfyuiIcLight").is_dir()

    def test_manifest_reports_installed_nodes(self, sofia):
        manifest = sofia.provision()
        assert [(n.name, n.status) for n in manifest.nodes] == [
            ("comfyui-kjnodes", "installed"),
            ("comfyui-ic-light", "installed"),
        ]

    def test_unpinned_node_installs_but_warns_about_reproducibility(self, worker):
        worker.environ["CUSTOM_NODES"] = "comfyui-kjnodes"
        manifest = worker.provision()
        assert worker.node_runner.install_calls() == ["comfyui-kjnodes"]
        assert any("pin" in w.lower() for w in manifest.warnings)

    def test_a_node_already_baked_into_the_image_is_not_reinstalled(self, worker):
        worker.bake_node("ComfyUI-KJNodes")  # baked dir names differ in casing
        worker.environ["CUSTOM_NODES"] = "comfyui-kjnodes@1.1.2"
        manifest = worker.provision()
        assert worker.node_runner.install_calls() == []
        assert manifest.nodes[0].status == "already-present"

    def test_works_without_a_network_volume(self, worker_no_volume):
        worker_no_volume.environ["CUSTOM_NODES"] = "comfyui-kjnodes@1.1.2"
        manifest = worker_no_volume.provision()
        assert manifest.nodes[0].status == "installed"
        assert (worker_no_volume.comfy_home / "custom_nodes/ComfyuiKjnodes").is_dir()


class TestVolumeCache:
    def test_first_boot_populates_the_cache(self, sofia):
        sofia.provision()
        cache_root = sofia.volume_path / CACHE_DIR_NAME
        (cache_key_dir,) = list(cache_root.iterdir())
        assert (cache_key_dir / ".complete").is_file()
        assert (cache_key_dir / "custom_nodes/ComfyuiKjnodes").is_dir()

    def test_second_boot_hits_the_cache_and_runs_no_installer(self, sofia):
        sofia.provision()
        # Simulate a fresh container on the same volume: wipe the
        # image-local custom_nodes tree, keep the volume.
        for node_dir in (sofia.comfy_home / "custom_nodes").iterdir():
            shutil.rmtree(node_dir)
        sofia.node_runner.calls.clear()

        manifest = sofia.provision()
        assert sofia.node_runner.install_calls() == []
        assert all(n.status == "cache-hit" for n in manifest.nodes)
        assert (sofia.comfy_home / "custom_nodes/ComfyuiKjnodes").is_dir()

    def test_python_dependencies_are_cached_and_exported_on_cache_hit(self, sofia):
        sofia.node_runner.requirements["comfyui-kjnodes"] = "einops>=0.6\n"
        sofia.provision()

        # pip deps were snapshotted into the cache with --target
        pip_calls = [c for c in sofia.node_runner.calls if c[:4] == ["python", "-m", "pip", "install"]]
        assert len(pip_calls) == 1
        assert "--target" in pip_calls[0]

        # A fresh container hitting the cache exports the dep dir on PYTHONPATH.
        for node_dir in (sofia.comfy_home / "custom_nodes").iterdir():
            shutil.rmtree(node_dir)
        sofia.provision()
        env_content = sofia.env_file.read_text()
        assert "PYTHONPATH" in env_content
        assert CACHE_DIR_NAME in env_content

    def test_changing_the_node_set_misses_the_cache(self, sofia):
        sofia.provision()
        sofia.node_runner.calls.clear()
        sofia.environ["CUSTOM_NODES"] += ", comfyui-easy-use@1.2.0"
        for node_dir in (sofia.comfy_home / "custom_nodes").iterdir():
            shutil.rmtree(node_dir)

        sofia.provision()
        assert "comfyui-easy-use@1.2.0" in sofia.node_runner.install_calls()

    def test_a_comfyui_version_bump_invalidates_the_cache(self, sofia):
        sofia.environ["COMFYUI_VERSION"] = "0.29.0"
        sofia.provision()
        first_keys = {p.name for p in (sofia.volume_path / CACHE_DIR_NAME).iterdir()}

        sofia.environ["COMFYUI_VERSION"] = "0.30.0"
        for node_dir in (sofia.comfy_home / "custom_nodes").iterdir():
            shutil.rmtree(node_dir)
        sofia.node_runner.calls.clear()
        sofia.provision()

        second_keys = {p.name for p in (sofia.volume_path / CACHE_DIR_NAME).iterdir()}
        assert second_keys != first_keys
        assert sofia.node_runner.install_calls() != []  # reinstalled, not reused

    def test_cache_completeness_marker_records_the_node_set(self, sofia):
        sofia.provision()
        cache_root = sofia.volume_path / CACHE_DIR_NAME
        (cache_key_dir,) = list(cache_root.iterdir())
        marker = json.loads((cache_key_dir / ".complete").read_text())
        assert marker["nodes"] == ["comfyui-kjnodes@1.1.2", "comfyui-ic-light@1.0.5"]


class TestSilentInstallFailures:
    """comfy-node-install's log parsing can report success for a node id
    that doesn't exist (found live: 'comfyui-gguf' vs the case-sensitive
    registry id 'ComfyUI-GGUF'). The filesystem is the truth: no new node
    directory after a "successful" install is a hard error."""

    def test_success_exit_without_node_dir_is_an_error(self, worker):
        # Runner returns success but creates nothing.
        class LyingRunner:
            def __init__(self, inner): self.inner = inner
            def __getattr__(self, item): return getattr(self.inner, item)
            def __call__(self, argv, **kw):
                from tests.conftest import FakeProc
                self.inner.calls.append(list(argv))
                return FakeProc()  # success, no side effects

        worker.environ["CUSTOM_NODES"] = "comfyui-gguf@1.0.0"
        import pytest as _pytest
        from provisioning import NodeInstallError
        with _pytest.raises(NodeInstallError) as excinfo:
            worker.provision(node_runner=LyingRunner(worker.node_runner))
        message = str(excinfo.value)
        assert "comfyui-gguf" in message
        assert "case-sensitive" in message


class TestInstallFailures:
    def test_a_bad_node_name_fails_with_registry_guidance(self, worker):
        worker.node_runner.failing_nodes.add("comfyui-kjnodez")  # typo
        worker.environ["CUSTOM_NODES"] = "comfyui-kjnodez@1.1.2"
        with pytest.raises(NodeInstallError) as excinfo:
            worker.provision()
        message = str(excinfo.value)
        assert "comfyui-kjnodez" in message
        assert "registry.comfy.org" in message

    def test_a_failed_install_never_writes_a_complete_cache(self, worker):
        worker.node_runner.failing_nodes.add("comfyui-broken")
        worker.environ["CUSTOM_NODES"] = "comfyui-kjnodes@1.1.2, comfyui-broken@1.0.0"
        with pytest.raises(NodeInstallError):
            worker.provision()
        cache_root = worker.volume_path / CACHE_DIR_NAME
        if cache_root.exists():
            for key_dir in cache_root.iterdir():
                assert not (key_dir / ".complete").exists()

    def test_nodes_install_even_when_no_models_are_configured(self, worker):
        worker.environ["CUSTOM_NODES"] = "comfyui-kjnodes@1.1.2"
        manifest = worker.provision()
        assert manifest.nodes and not manifest.models
        assert worker.session.requests == []  # no model traffic
