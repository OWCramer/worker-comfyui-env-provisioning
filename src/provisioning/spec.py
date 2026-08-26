"""Parse the runtime provisioning contract from environment variables.

Users configure custom models and custom nodes on their endpoint with plain
environment variables — no Docker build required:

    CHECKPOINT_URLS = https://civitai.com/models/4384/dreamshaper
    LORA_URLS       = https://civitai.com/api/download/models/87153::add_detail.safetensors
    CUSTOM_NODES    = comfyui-kjnodes@1.1.2, comfyui-ic-light

Model entries are separated by commas or newlines. An entry may carry an
explicit target filename with the ``url::filename`` syntax; otherwise the
filename is resolved from the source (Civitai API, Hugging Face URL path,
or the URL basename).

Custom node entries are Comfy Registry ids, optionally pinned as
``name@version``.
"""

import re
from dataclasses import dataclass, field

from .errors import ConfigError

# Environment variable → subdirectory of the ComfyUI ``models/`` tree.
MODEL_ENV_VARS = {
    "CHECKPOINT_URLS": "checkpoints",
    "LORA_URLS": "loras",
    "VAE_URLS": "vae",
    "CONTROLNET_URLS": "controlnet",
    "UPSCALE_MODEL_URLS": "upscale_models",
    "EMBEDDING_URLS": "embeddings",
    "CLIP_URLS": "clip",
    "CLIP_VISION_URLS": "clip_vision",
    "DIFFUSION_MODEL_URLS": "diffusion_models",
    "TEXT_ENCODER_URLS": "text_encoders",
    "UNET_URLS": "unet",
    # Impact Pack / detailer ecosystems read from these subtrees:
    "BBOX_MODEL_URLS": "ultralytics/bbox",
    "SEGM_MODEL_URLS": "ultralytics/segm",
    "SAM_MODEL_URLS": "sams",
}

CUSTOM_NODES_ENV_VAR = "CUSTOM_NODES"
KILL_SWITCH_ENV_VAR = "RUNTIME_PROVISIONING"

_NODE_PART_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
_URL_RE = re.compile(r"^https?://\S+$")


@dataclass(frozen=True)
class ModelSpec:
    url: str
    directory: str  # subdirectory of models/, e.g. "checkpoints"
    env_var: str  # which env var this entry came from (for error messages)
    filename: str | None = None  # explicit override via ``url::filename``


@dataclass(frozen=True)
class NodeSpec:
    name: str
    version: str | None = None

    def __str__(self):
        return f"{self.name}@{self.version}" if self.version else self.name


@dataclass
class ProvisionPlan:
    models: list[ModelSpec] = field(default_factory=list)
    nodes: list[NodeSpec] = field(default_factory=list)

    @property
    def is_empty(self):
        return not self.models and not self.nodes


def provisioning_enabled(environ):
    """The RUNTIME_PROVISIONING kill switch. Enabled unless explicitly off."""
    value = environ.get(KILL_SWITCH_ENV_VAR, "true").strip().lower()
    return value not in ("false", "0", "no", "off")


def _split_entries(raw):
    entries = []
    for chunk in re.split(r"[,\n]", raw):
        chunk = chunk.strip()
        if chunk:
            entries.append(chunk)
    return entries


def _valid_filename(name):
    if not name or name in (".", ".."):
        return False
    if "/" in name or "\\" in name:
        return False
    return True


def _parse_model_entry(entry, env_var, directory, problems):
    filename = None
    url = entry
    if "::" in entry:
        url, _, filename = entry.rpartition("::")
        filename = filename.strip()
        url = url.strip()
        if not _valid_filename(filename):
            problems.append(
                f"{env_var}: invalid filename {filename!r} in entry {entry!r} "
                "(filenames must not contain path separators)"
            )
            return None
    if not _URL_RE.match(url):
        problems.append(
            f"{env_var}: {url!r} is not a valid http(s) URL "
            "(use 'https://<url>' or 'https://<url>::<filename>')"
        )
        return None
    return ModelSpec(url=url, directory=directory, env_var=env_var, filename=filename)


def _parse_node_entry(entry, problems):
    name, sep, version = entry.partition("@")
    name = name.strip()
    version = version.strip() if sep else None
    if not _NODE_PART_RE.match(name) or (version is not None and not _NODE_PART_RE.match(version)):
        problems.append(
            f"{CUSTOM_NODES_ENV_VAR}: invalid entry {entry!r} "
            "(expected a Comfy Registry id, optionally pinned as 'name@version')"
        )
        return None
    return NodeSpec(name=name, version=version)


def parse_plan(environ):
    """Build a ProvisionPlan from environment variables.

    Raises ConfigError listing *every* invalid entry, so users can fix
    their endpoint configuration in one pass.
    """
    problems = []
    plan = ProvisionPlan()

    for env_var, directory in MODEL_ENV_VARS.items():
        raw = environ.get(env_var, "")
        for entry in _split_entries(raw):
            spec = _parse_model_entry(entry, env_var, directory, problems)
            if spec is not None:
                plan.models.append(spec)

    for entry in _split_entries(environ.get(CUSTOM_NODES_ENV_VAR, "")):
        spec = _parse_node_entry(entry, problems)
        if spec is not None:
            plan.nodes.append(spec)

    if problems:
        raise ConfigError(
            "Invalid provisioning configuration:\n  - " + "\n  - ".join(problems)
        )

    # De-duplicate while preserving order (same URL listed twice, etc.)
    plan.models = list(dict.fromkeys(plan.models))
    plan.nodes = list(dict.fromkeys(plan.nodes))
    return plan
