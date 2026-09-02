"""Work out what a Hugging Face model is, and everything needed to run it.

The console names one repository. Everything else — which file in it holds the
weights, whether it is a checkpoint or a LoRA, which workflow template fits, and
what else has to be downloaded alongside — is decided here, next to the
templates that consume it.

Hugging Face's own metadata answers most of it: the `lora` tag and
`base_model:adapter:` say a repository is an adapter, and `diffusers:<Pipeline>`
names the architecture outright, which is how any fine-tune of a family we
support is recognised without knowing its name.
"""

import re
from dataclasses import dataclass, field

from . import known_models
from .errors import ResolutionError
from .spec import ModelSpec

HF_API = "https://huggingface.co/api/models"

# A shard cannot be loaded as a single-file checkpoint.
SHARD_RE = re.compile(r"-\d{5}-of-\d{5}\.")
PIPELINE_TAG_RE = re.compile(r"^diffusers:(\w+)$")
COMPANION_TOKENS = {"ae", "vae", "clip", "encoder", "lora", "loras"}


@dataclass
class Detected:
    template: str
    models: list = field(default_factory=list)  # ModelSpec
    warnings: list = field(default_factory=list)


def _tokens(filename):
    return re.split(r"[_\-.\s]+", filename.rsplit(".safetensors", 1)[0].lower())


def _is_companion(filename):
    if any(token in COMPANION_TOKENS for token in _tokens(filename)):
        return True
    return "text_encoder" in filename.lower()


def parse_source(source):
    """Split what the console gave us into ``(repo_id, file_path_or_None)``."""
    value = source.strip()
    if not value:
        raise ResolutionError("COMFY_MODEL is empty.")

    if "://" not in value:
        if value.count("/") != 1:
            raise ResolutionError(
                f"COMFY_MODEL {value!r} is not a Hugging Face repository. "
                "Use 'owner/name', or a link to a file in one."
            )
        return value, None

    match = re.match(
        r"^https?://(?:www\.)?huggingface\.co/([^/]+)/([^/]+)"
        r"(?:/(?:resolve|blob)/[^/]+/(.+))?/?$",
        value,
    )
    if not match:
        raise ResolutionError(
            f"COMFY_MODEL {value!r} is not a Hugging Face link. "
            "Only Hugging Face models are detected automatically."
        )
    owner, name, path = match.groups()
    return f"{owner}/{name}", path


def fetch_repo(repo_id, *, session, hf_token):
    headers = {"Authorization": f"Bearer {hf_token}"} if hf_token else {}
    response = session.get(f"{HF_API}/{repo_id}?blobs=true", headers=headers)
    if response.status_code == 404:
        raise ResolutionError(f"Hugging Face has no model {repo_id!r}.")
    if response.status_code in (401, 403):
        raise ResolutionError(
            f"{repo_id} is gated. Set HF_TOKEN to a token from an account that "
            "has accepted its licence."
        )
    if response.status_code != 200:
        raise ResolutionError(
            f"Hugging Face returned {response.status_code} for {repo_id}."
        )
    try:
        return response.json()
    except ValueError as exc:
        raise ResolutionError(f"Hugging Face sent an unreadable reply for {repo_id}: {exc}") from exc


def pipeline_class(repo):
    for tag in repo.get("tags") or []:
        match = PIPELINE_TAG_RE.match(tag)
        if match:
            return match.group(1)
    return None


def is_lora(repo):
    tags = repo.get("tags") or []
    return any(tag == "lora" or tag.startswith("base_model:adapter:") for tag in tags)


def base_model_label(repo):
    declared = (repo.get("cardData") or {}).get("base_model")
    if isinstance(declared, list):
        return declared[0] if declared else None
    if declared:
        return declared
    for tag in repo.get("tags") or []:
        if tag.startswith("base_model:"):
            return tag.rsplit(":", 1)[-1]
    return None


def weight_candidates(repo, *, nested_prefix=None):
    """Files that could be the weights, companions and shards excluded."""
    names = [
        sibling["rfilename"]
        for sibling in repo.get("siblings") or []
        if sibling.get("rfilename", "").endswith(".safetensors")
    ]
    names = [name for name in names if not SHARD_RE.search(name)]
    if nested_prefix:
        names = [n for n in names if n.startswith(nested_prefix) or "/" not in n]
    else:
        names = [n for n in names if "/" not in n]

    weights = [n for n in names if not _is_companion(n.rsplit("/", 1)[-1])]
    return weights or names


def pick_weights(repo, *, repo_id, nested_prefix=None):
    candidates = weight_candidates(repo, nested_prefix=nested_prefix)
    if not candidates:
        raise ResolutionError(
            f"{repo_id} publishes no single-file .safetensors we can load. "
            "Link the file directly, or use a repository that ships one."
        )
    if len(candidates) == 1:
        return candidates[0]

    # Tie-breaks, in the order that has proved right: a plain build over one with
    # a VAE folded in, then the lower-precision build.
    plain = [n for n in candidates if "vae" not in n.lower()]
    if len(plain) == 1:
        return plain[0]
    fp8 = [n for n in candidates if "fp8" in n.lower()]
    if len(fp8) == 1:
        return fp8[0]

    raise ResolutionError(
        f"{repo_id} publishes several builds ({', '.join(sorted(candidates)[:6])}"
        f"{'…' if len(candidates) > 6 else ''}). "
        "Point COMFY_MODEL at the one you want."
    )


def _spec(url, directory, filename):
    return ModelSpec(url=url, directory=directory, env_var="COMFY_MODEL", filename=filename)


def _hf_url(repo_id, path):
    return f"https://huggingface.co/{repo_id}/resolve/main/{path}"


WAN_RE = re.compile(r"(^|[/_\s.-])wan[\s._-]?\d|(^|[/_\s.-])wan([/_\s.-]|$)", re.I)
TI2V_RE = re.compile(r"ti2v", re.I)
TWO_STAGE_RE = re.compile(r"high[_\s-]?noise|low[_\s-]?noise", re.I)


def _wan_template(repo_id, filename, tags):
    """Which Wan pipeline this is, since they need different templates."""
    joined = " ".join([repo_id, filename or "", *(tags or [])])
    if TI2V_RE.search(joined):
        return "wan-ti2v"
    if TWO_STAGE_RE.search(joined):
        raise ResolutionError(
            f"{repo_id} is a two-stage Wan model (a high-noise and a low-noise "
            "pass), which no template here covers yet."
        )
    if re.search(r"wan[\s._-]?2\.1", joined, re.I):
        return "wan-t2v"
    raise ResolutionError(
        f"{repo_id} is a Wan model we have no template for. Wan 2.2 TI2V and "
        "Wan 2.1 are supported."
    )


def _detect_video(repo, repo_id, requested_file):
    filename = requested_file or pick_weights(
        repo, repo_id=repo_id, nested_prefix="split_files/diffusion_models/"
    )
    template = _wan_template(repo_id, filename, repo.get("tags"))
    companions = known_models.WAN_COMPANIONS[template]

    return Detected(
        template=template,
        models=[
            _spec(_hf_url(repo_id, filename), "diffusion_models", "model.safetensors"),
            _spec(companions["text_encoders"], "text_encoders", "text_encoder.safetensors"),
            _spec(companions["vae"], "vae", "vae.safetensors"),
        ],
    )


def _detect_lora(repo, repo_id, requested_file, *, session, hf_token):
    filename = requested_file or pick_weights(repo, repo_id=repo_id)
    label = base_model_label(repo)

    pipeline = None
    if label and re.fullmatch(r"[\w.-]+/[\w.-]+", label):
        try:
            pipeline = pipeline_class(fetch_repo(label, session=session, hf_token=hf_token))
        except ResolutionError:
            pipeline = None

    base = known_models.BASE_CHECKPOINTS.get(pipeline or "")
    if base is None:
        raise ResolutionError(
            f"{repo_id} is a LoRA trained on {label or 'an unrecognised checkpoint'}, "
            "which no template here covers. Set COMFY_TEMPLATE and CHECKPOINT_URLS "
            "yourself to stack it on a checkpoint of your choosing."
        )

    return Detected(
        template=f"{base['template']}-lora",
        models=[
            _spec(base["url"], "checkpoints", "model.safetensors"),
            _spec(_hf_url(repo_id, filename), "loras", "lora.safetensors"),
        ],
        warnings=[
            f"{repo_id} is a LoRA; stacking it on {base['url'].rsplit('/', 1)[-1]} "
            f"because it declares {label}."
        ],
    )


def detect(source, *, session, hf_token=None):
    """Everything the endpoint needs in order to run ``source``."""
    repo_id, requested_file = parse_source(source)
    repo = fetch_repo(repo_id, session=session, hf_token=hf_token)

    if (repo.get("pipeline_tag") or "").endswith("-to-video") or WAN_RE.search(repo_id):
        return _detect_video(repo, repo_id, requested_file)

    if is_lora(repo):
        return _detect_lora(repo, repo_id, requested_file, session=session, hf_token=hf_token)

    filename = requested_file or pick_weights(repo, repo_id=repo_id)
    template = known_models.CHECKPOINT_TEMPLATES.get(pipeline_class(repo) or "")
    warnings = []
    if template is None:
        # CheckpointLoaderSimple loads most single-file checkpoints whatever the
        # architecture, so an unfamiliar one is worth trying with a warning
        # rather than refusing outright.
        template = "checkpoint"
        warnings.append(
            f"{repo_id} declares no architecture we recognise; running it on the "
            "checkpoint template. Set COMFY_TEMPLATE if that is wrong."
        )

    return Detected(
        template=template,
        models=[_spec(_hf_url(repo_id, filename), "checkpoints", "model.safetensors")],
        warnings=warnings,
    )
