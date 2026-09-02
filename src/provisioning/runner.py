"""Orchestrate startup provisioning: custom nodes first, then models.

Called from start.sh (``python -m provisioning``) before ComfyUI launches.
Everything is driven by environment variables — see spec.py for the contract.
A failure here fails the worker fast with an actionable message, which beats
booting a worker whose workflows 404 on missing models.
"""

import json
import os
import time
from dataclasses import dataclass, field

from . import download, hf_cache, nodes, resolve, spec


def _log(message):
    print(f"worker-comfyui (provisioning): {message}", flush=True)


@dataclass
class Manifest:
    models: list = field(default_factory=list)  # DownloadResult
    nodes: list = field(default_factory=list)  # NodeResult
    warnings: list = field(default_factory=list)
    enabled: bool = True

    @property
    def is_noop(self):
        return not self.models and not self.nodes

    def to_dict(self):
        return {
            "enabled": self.enabled,
            "models": [vars(m) for m in self.models],
            "nodes": [vars(n) for n in self.nodes],
            "warnings": list(self.warnings),
        }


def _hub_fetcher(cache_dir, hf_token):
    """Bind the shared cache and token to a per-file fetch, or None without a volume."""
    if cache_dir is None:
        return None

    def fetch(repo_id, filename, *, revision):
        return hf_cache.hub_download(
            repo_id, filename, revision=revision, token=hf_token, cache_dir=cache_dir
        )

    return fetch


def _default_session():
    import requests

    return requests.Session()


def provision(
    environ=None,
    *,
    session=None,
    comfy_home="/comfyui",
    volume_path="/runpod-volume",
    env_file="/tmp/provision_env.sh",
    manifest_path="/tmp/provision_manifest.json",
    node_runner=None,
    hub=None,
    sleep=time.sleep,
    lock_wait_seconds=1800.0,
    lock_stale_seconds=3600.0,
):
    environ = os.environ if environ is None else environ
    manifest = Manifest()

    if not spec.provisioning_enabled(environ):
        _log("RUNTIME_PROVISIONING is disabled — skipping.")
        manifest.enabled = False
        return manifest

    plan = spec.parse_plan(environ)
    if plan.is_empty:
        return manifest

    if plan.nodes:
        import subprocess

        _log(f"Installing {len(plan.nodes)} custom node(s): "
             + ", ".join(str(n) for n in plan.nodes))
        manifest.nodes = nodes.install_nodes(
            plan.nodes,
            comfy_home=comfy_home,
            volume_path=volume_path,
            comfyui_version=environ.get("COMFYUI_VERSION", "unknown"),
            env_file=env_file,
            runner=node_runner if node_runner is not None else subprocess.run,
            warnings=manifest.warnings,
        )

    if plan.models:
        session = session if session is not None else _default_session()
        civitai_token = environ.get("CIVITAI_TOKEN") or environ.get("CIVITAI_API_TOKEN")
        hf_token = environ.get("HF_TOKEN") or environ.get("HUGGINGFACE_ACCESS_TOKEN")
        root, on_volume = download.models_root(comfy_home, volume_path)
        hub = _hub_fetcher(hf_cache.cache_dir(volume_path), hf_token) if hub is None else hub
        _log(
            f"Fetching {len(plan.models)} model(s) into {root}"
            + (" (network volume)" if on_volume else "")
        )
        if hub is not None:
            _log("Using the shared Hugging Face cache on the volume.")
        for model_spec in plan.models:
            resolved = resolve.resolve_model(
                model_spec,
                session=session,
                civitai_token=civitai_token,
                hf_token=hf_token,
            )
            manifest.warnings.extend(resolved.warnings)
            result = download.download_model(
                resolved,
                model_spec,
                root=root,
                session=session,
                hub=hub,
                sleep=sleep,
                lock_wait_seconds=lock_wait_seconds,
                lock_stale_seconds=lock_stale_seconds,
            )
            _log(
                f"  {result.status}: models/{result.directory}/{result.filename}"
            )
            manifest.models.append(result)

    for warning in manifest.warnings:
        _log(f"WARNING: {warning}")

    if not manifest.is_noop:
        with open(manifest_path, "w") as fh:
            json.dump(manifest.to_dict(), fh, indent=2)
        _summarize(manifest)

    return manifest


def _summarize(manifest):
    if manifest.nodes:
        _log("Custom nodes ready:")
        for n in manifest.nodes:
            pin = f"@{n.version}" if n.version else ""
            _log(f"  - {n.name}{pin} ({n.status})")
    if manifest.models:
        _log("Models ready — reference them in workflows by these filenames:")
        for m in manifest.models:
            _log(f"  - {m.filename}  (models/{m.directory}, {m.status})")
