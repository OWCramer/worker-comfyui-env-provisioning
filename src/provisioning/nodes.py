"""Install Comfy Registry custom nodes at worker startup.

Nodes come from the CUSTOM_NODES environment variable as registry ids,
optionally pinned (``comfyui-kjnodes@1.1.2``). Installation uses the same
``comfy-node-install`` wrapper the Dockerfile path uses, so behavior is
identical to a baked image — just resolved at startup instead of build time.

When a network volume is mounted, installs are cached under
``<volume>/.worker-comfyui/node-cache/<key>/`` where the key hashes the
ComfyUI version and the exact node set:

- cache miss: install normally (deps land in the image venv for this worker),
  then snapshot the new ``custom_nodes/`` dirs and a ``pip --target`` copy of
  their Python deps into the cache.
- cache hit: copy node dirs out of the cache and export the cached dep dir on
  PYTHONPATH via an env file that start.sh sources — no network, no pip.

The cache key includes the ComfyUI version so a base-image upgrade never
reuses deps compiled against an older environment.
"""

import hashlib
import json
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

from .errors import NodeInstallError

CACHE_DIR_NAME = ".worker-comfyui/node-cache"
COMPLETE_MARKER = ".complete"


@dataclass
class NodeResult:
    name: str
    version: str | None
    status: str  # "installed" | "cache-hit" | "already-present"


def _normalize(name):
    """Registry ids and repo dir names differ in case/punctuation."""
    return "".join(c for c in name.lower() if c.isalnum())


def _existing_node_dirs(custom_nodes_dir):
    if not custom_nodes_dir.is_dir():
        return set()
    return {p.name for p in custom_nodes_dir.iterdir() if p.is_dir()}


def cache_key(specs, comfyui_version):
    digest = hashlib.sha256(
        "|".join([comfyui_version] + sorted(str(s) for s in specs)).encode()
    )
    return digest.hexdigest()[:16]


def _run(runner, argv, error_context):
    proc = runner(argv, capture_output=True, text=True)
    if proc.returncode != 0:
        output = (proc.stderr or proc.stdout or "").strip()
        raise NodeInstallError(f"{error_context}:\n{output}")
    return proc


def _restore_from_cache(cache_dir, custom_nodes_dir, env_file, warnings):
    restored = []
    cached_nodes = cache_dir / "custom_nodes"
    for node_dir in sorted(cached_nodes.iterdir()):
        if not node_dir.is_dir():
            continue
        target = custom_nodes_dir / node_dir.name
        if not target.exists():
            shutil.copytree(node_dir, target)
        restored.append(node_dir.name)

    pip_dir = cache_dir / "pip"
    if pip_dir.is_dir() and any(pip_dir.iterdir()):
        Path(env_file).write_text(
            f'export PYTHONPATH="{pip_dir}:${{PYTHONPATH:-}}"\n'
        )
    return restored


def _populate_cache(cache_dir, custom_nodes_dir, new_dirs, specs, runner):
    cached_nodes = cache_dir / "custom_nodes"
    cached_nodes.mkdir(parents=True, exist_ok=True)
    pip_dir = cache_dir / "pip"

    for name in sorted(new_dirs):
        source = custom_nodes_dir / name
        target = cached_nodes / name
        if source.is_dir() and not target.exists():
            shutil.copytree(source, target)
        requirements = source / "requirements.txt"
        if requirements.is_file():
            pip_dir.mkdir(parents=True, exist_ok=True)
            _run(
                runner,
                ["python", "-m", "pip", "install", "-r", str(requirements),
                 "--target", str(pip_dir)],
                f"Failed to cache dependencies for custom node '{name}'",
            )

    (cache_dir / COMPLETE_MARKER).write_text(
        json.dumps({"nodes": [str(s) for s in specs]})
    )


def install_nodes(
    specs,
    *,
    comfy_home,
    volume_path,
    comfyui_version,
    env_file,
    runner=subprocess.run,
    warnings=None,
):
    """Install the requested custom nodes, using the volume cache if possible."""
    warnings = warnings if warnings is not None else []
    custom_nodes_dir = Path(comfy_home) / "custom_nodes"
    custom_nodes_dir.mkdir(parents=True, exist_ok=True)

    results = []
    existing_normalized = {_normalize(d) for d in _existing_node_dirs(custom_nodes_dir)}
    to_install = []
    for spec in specs:
        if _normalize(spec.name) in existing_normalized:
            results.append(NodeResult(spec.name, spec.version, "already-present"))
        else:
            to_install.append(spec)
            if spec.version is None:
                warnings.append(
                    f"CUSTOM_NODES: '{spec.name}' is not pinned to a version — "
                    "workers scaling up later may install a newer release. "
                    f"Pin it as '{spec.name}@<version>' for reproducible deploys."
                )

    if not to_install:
        return results

    volume = Path(volume_path)
    cache_dir = None
    if volume.is_dir():
        cache_dir = volume / CACHE_DIR_NAME / cache_key(to_install, comfyui_version)

    if cache_dir is not None and (cache_dir / COMPLETE_MARKER).is_file():
        _restore_from_cache(cache_dir, custom_nodes_dir, env_file, warnings)
        results.extend(
            NodeResult(s.name, s.version, "cache-hit") for s in to_install
        )
        return results

    before = _existing_node_dirs(custom_nodes_dir)
    for spec in to_install:
        _run(
            runner,
            ["comfy-node-install", str(spec)],
            f"Failed to install custom node '{spec}' — verify the name (and "
            "version) at https://registry.comfy.org/",
        )
        # comfy-node-install's log parsing can miss failures (e.g. an id that
        # doesn't exist reports success). Trust the filesystem, not the exit
        # code: a real install always creates a node directory.
        after = _existing_node_dirs(custom_nodes_dir)
        installed_normalized = {_normalize(d) for d in after}
        if not (after - before) and _normalize(spec.name) not in installed_normalized:
            raise NodeInstallError(
                f"comfy-node-install reported success for '{spec}' but no node "
                "directory appeared. The registry id is likely wrong — ids are "
                "case-sensitive (e.g. 'ComfyUI-GGUF', not 'comfyui-gguf'); "
                "verify at https://registry.comfy.org/"
            )
        results.append(NodeResult(spec.name, spec.version, "installed"))
    new_dirs = _existing_node_dirs(custom_nodes_dir) - before

    if cache_dir is not None:
        _populate_cache(cache_dir, custom_nodes_dir, new_dirs, to_install, runner)

    return results
