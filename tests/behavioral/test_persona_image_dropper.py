"""Persona: Ana — shares her generated images and expects them to round-trip.

Ana generates an image on her endpoint, sends the PNG to a friend, and the
friend drags it into ComfyUI (or a future Runpod deploy flow) to get the full
workflow back. That round-trip only works if the worker embeds workflow
metadata in saved images — ComfyUI's own default, which this worker
historically stripped with a hardcoded ``--disable-metadata``.

Contract: EMBED_WORKFLOW_METADATA (default on, kill switch off) controls the
flag. These tests exercise ``src/launch_flags.sh`` directly, the sourceable
helper start.sh uses to build the ComfyUI launch command.
"""

import subprocess
from pathlib import Path

import pytest

LAUNCH_FLAGS_SH = Path(__file__).resolve().parents[2] / "src" / "launch_flags.sh"


def launch_flags(**env):
    exports = "".join(f'export {k}="{v}"\n' for k, v in env.items())
    script = f"{exports}source {LAUNCH_FLAGS_SH}\ncomfy_launch_flags"
    result = subprocess.run(
        ["bash", "-c", script], capture_output=True, text=True, check=True
    )
    return result.stdout.strip().split()


class TestMetadataEmbedding:
    def test_metadata_is_embedded_by_default(self):
        assert "--disable-metadata" not in launch_flags()

    def test_explicit_true_embeds_metadata(self):
        assert "--disable-metadata" not in launch_flags(EMBED_WORKFLOW_METADATA="true")

    @pytest.mark.parametrize("value", ["false", "FALSE", "0", "no", "off"])
    def test_kill_switch_strips_metadata(self, value):
        assert "--disable-metadata" in launch_flags(EMBED_WORKFLOW_METADATA=value)

    def test_unrecognized_values_keep_the_default(self):
        assert "--disable-metadata" not in launch_flags(
            EMBED_WORKFLOW_METADATA="banana"
        )


class TestOtherLaunchFlags:
    """The helper carries the pre-existing launch behavior unchanged."""

    def test_baseline_flags_are_always_present(self):
        flags = launch_flags()
        assert "--disable-auto-launch" in flags
        assert "--log-stdout" in flags

    def test_log_level_defaults_to_debug(self):
        flags = launch_flags()
        assert flags[flags.index("--verbose") + 1] == "DEBUG"

    def test_log_level_is_configurable(self):
        flags = launch_flags(COMFY_LOG_LEVEL="INFO")
        assert flags[flags.index("--verbose") + 1] == "INFO"

    def test_listen_only_in_local_api_mode(self):
        assert "--listen" not in launch_flags()
        assert "--listen" in launch_flags(SERVE_API_LOCALLY="true")
