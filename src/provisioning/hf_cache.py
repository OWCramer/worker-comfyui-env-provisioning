"""Fetch Hugging Face files through the shared cache on the network volume.

Runpod mounts a Hugging Face cache at ``<volume>/huggingface-cache/hub`` and
pre-populates it with the repository named in the endpoint's Model field, at no
charge to the user. ``worker-vllm`` consumes it by pointing the Hugging Face
libraries at that directory and letting them decide, and this does the same:

- a file already in the cache is used as-is, so the first request does not wait
  for a download it has effectively already paid for;
- a file that is not is downloaded *into* the cache, so the next worker — and
  any other endpoint on the same volume — gets it for free. That matters because
  the platform pre-caches only one repository per endpoint, while a video model
  needs three.

The cache layout is Hugging Face's own, so ``huggingface_hub`` handles revisions
and integrity rather than this module guessing at paths.
"""

from pathlib import Path
from urllib.parse import unquote, urlparse

CACHE_DIR_NAME = "huggingface-cache"


def cache_dir(volume_path):
    """The shared hub cache, when a network volume is mounted to hold it."""
    volume = Path(volume_path)
    if not volume.is_dir():
        return None
    return volume / CACHE_DIR_NAME / "hub"


def parse_repo_file(url):
    """Split a Hugging Face file URL into ``(repo_id, path_within_repo, revision)``.

    Returns ``None`` for anything else, including bare repository links, which
    name no file to fetch.
    """
    parsed = urlparse(url)
    if parsed.hostname not in ("huggingface.co", "www.huggingface.co"):
        return None

    segments = [unquote(part) for part in parsed.path.split("/") if part]
    # <owner>/<name>/resolve/<revision>/<path...>
    if len(segments) < 5 or segments[2] not in ("resolve", "blob"):
        return None

    return f"{segments[0]}/{segments[1]}", "/".join(segments[4:]), segments[3]


def hub_download(repo_id, filename, *, revision, token, cache_dir):
    """Resolve a file through the shared cache, downloading it only if absent.

    Imported lazily so the module stays usable — and testable — in an
    environment without ``huggingface_hub`` installed.
    """
    from huggingface_hub import hf_hub_download

    return hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        revision=revision,
        token=token or None,
        cache_dir=str(cache_dir),
    )
