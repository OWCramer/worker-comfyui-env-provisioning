"""Resolve model source URLs into concrete downloads.

Supported sources:

- Civitai model pages:      https://civitai.com/models/<id>[/slug][?modelVersionId=N]
- Civitai download links:   https://civitai.com/api/download/models/<versionId>
- Hugging Face file links:  https://huggingface.co/<repo>/resolve/<rev>/<file>
- Any direct http(s) URL whose path ends in a filename

Civitai URLs are resolved through the public Civitai API so we learn the
canonical filename (workflows reference models by filename) and the model
type (so we can warn when e.g. a LoRA is configured as a checkpoint).
"""

import re
from dataclasses import dataclass, field
from urllib.parse import urlparse, parse_qs, urlencode, urlunparse

from .errors import ResolutionError

CIVITAI_MODEL_PAGE_RE = re.compile(r"civitai\.com/models/(\d+)")
CIVITAI_DOWNLOAD_RE = re.compile(r"civitai\.com/api/download/models/(\d+)")

# Civitai model types → the models/ subdirectory they belong in.
CIVITAI_TYPE_DIRS = {
    "Checkpoint": "checkpoints",
    "LORA": "loras",
    "LoCon": "loras",
    "DoRA": "loras",
    "TextualInversion": "embeddings",
    "VAE": "vae",
    "Controlnet": "controlnet",
    "Upscaler": "upscale_models",
}


@dataclass
class ResolvedModel:
    download_url: str
    filename: str
    source: str  # "civitai" | "huggingface" | "direct"
    headers: dict = field(default_factory=dict)
    warnings: list = field(default_factory=list)


def _civitai_headers(token):
    return {"Authorization": f"Bearer {token}"} if token else {}


def _civitai_api_get(session, url, token, what):
    response = session.get(url, headers=_civitai_headers(token))
    if response.status_code in (401, 403):
        raise ResolutionError(
            f"Civitai denied access to {what} ({response.status_code}). "
            "This model likely requires authentication — set the CIVITAI_TOKEN "
            "environment variable to a Civitai API token "
            "(https://civitai.com/user/account)."
        )
    if response.status_code == 404:
        raise ResolutionError(f"Civitai returned 404 for {what} — check the URL.")
    if response.status_code != 200:
        raise ResolutionError(
            f"Civitai API error {response.status_code} while resolving {what}."
        )
    try:
        return response.json()
    except ValueError as exc:
        raise ResolutionError(
            f"Civitai returned an unparseable response for {what}: {exc}"
        ) from exc


def _append_query_param(url, key, value):
    parsed = urlparse(url)
    query = parse_qs(parsed.query)
    query[key] = [value]
    return urlunparse(parsed._replace(query=urlencode(query, doseq=True)))


def _pick_file(version, what):
    files = version.get("files") or []
    if not files:
        raise ResolutionError(f"Civitai lists no downloadable files for {what}.")
    for f in files:
        if f.get("primary"):
            return f
    return files[0]


def _check_civitai_type(model_type, spec, warnings):
    expected_dir = CIVITAI_TYPE_DIRS.get(model_type)
    if expected_dir and expected_dir != spec.directory:
        warnings.append(
            f"{spec.env_var}: Civitai reports this model is a '{model_type}' "
            f"(belongs in models/{expected_dir}), but it is configured for "
            f"models/{spec.directory}. Downloading anyway — double-check your "
            "workflow if the model fails to load."
        )


def _resolve_civitai_page(spec, model_id, session, token):
    data = _civitai_api_get(
        session,
        f"https://civitai.com/api/v1/models/{model_id}",
        token,
        f"model {model_id}",
    )
    versions = data.get("modelVersions") or []
    if not versions:
        raise ResolutionError(f"Civitai model {model_id} has no versions.")

    requested = parse_qs(urlparse(spec.url).query).get("modelVersionId", [None])[0]
    version = versions[0]
    if requested:
        matches = [v for v in versions if str(v.get("id")) == str(requested)]
        if not matches:
            raise ResolutionError(
                f"Civitai model {model_id} has no version {requested} "
                f"(available: {', '.join(str(v.get('id')) for v in versions)})."
            )
        version = matches[0]

    file = _pick_file(version, f"model {model_id}")
    warnings = []
    _check_civitai_type(data.get("type"), spec, warnings)

    download_url = file.get("downloadUrl") or (
        f"https://civitai.com/api/download/models/{version['id']}"
    )
    if token:
        download_url = _append_query_param(download_url, "token", token)
    return ResolvedModel(
        download_url=download_url,
        filename=spec.filename or file["name"],
        source="civitai",
        headers=_civitai_headers(token),
        warnings=warnings,
    )


def _resolve_civitai_download(spec, version_id, session, token):
    data = _civitai_api_get(
        session,
        f"https://civitai.com/api/v1/model-versions/{version_id}",
        token,
        f"model version {version_id}",
    )
    file = _pick_file(data, f"model version {version_id}")
    warnings = []
    _check_civitai_type((data.get("model") or {}).get("type"), spec, warnings)

    download_url = spec.url
    if token:
        download_url = _append_query_param(download_url, "token", token)
    return ResolvedModel(
        download_url=download_url,
        filename=spec.filename or file["name"],
        source="civitai",
        headers=_civitai_headers(token),
        warnings=warnings,
    )


def _filename_from_path(url):
    path = urlparse(url).path
    name = path.rsplit("/", 1)[-1]
    return name or None


def _resolve_huggingface(spec, hf_token):
    filename = spec.filename or _filename_from_path(spec.url)
    if not filename:
        raise ResolutionError(
            f"{spec.env_var}: cannot derive a filename from {spec.url!r} — "
            "append '::<filename>' to the entry."
        )
    headers = {"Authorization": f"Bearer {hf_token}"} if hf_token else {}
    return ResolvedModel(
        download_url=spec.url, filename=filename, source="huggingface", headers=headers
    )


def _resolve_direct(spec):
    filename = spec.filename or _filename_from_path(spec.url)
    if not filename or "." not in filename:
        raise ResolutionError(
            f"{spec.env_var}: cannot derive a filename from {spec.url!r} — "
            "append '::<filename>' to the entry (e.g. "
            f"'{spec.url}::my_model.safetensors')."
        )
    return ResolvedModel(download_url=spec.url, filename=filename, source="direct")


def resolve_model(spec, *, session, civitai_token=None, hf_token=None):
    """Resolve a ModelSpec into a ResolvedModel ready for download."""
    match = CIVITAI_DOWNLOAD_RE.search(spec.url)
    if match:
        return _resolve_civitai_download(spec, match.group(1), session, civitai_token)
    match = CIVITAI_MODEL_PAGE_RE.search(spec.url)
    if match:
        return _resolve_civitai_page(spec, match.group(1), session, civitai_token)
    if "huggingface.co" in urlparse(spec.url).netloc:
        return _resolve_huggingface(spec, hf_token)
    return _resolve_direct(spec)
