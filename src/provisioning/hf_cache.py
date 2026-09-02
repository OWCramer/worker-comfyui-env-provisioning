"""Find models Runpod has already cached on the host.

An endpoint with a cached model gets that Hugging Face repository mounted under
``<volume>/huggingface-cache/hub`` in Hugging Face's own cache layout. The
platform populates it before the worker starts and does not bill for the time,
so linking a file out of it removes a download that would otherwise run inside
the user's first request.

The layout and the order of resolution follow Runpod's documented helper:
https://docs.runpod.io/serverless/development/huggingface-models#use-cached-models
"""

import os
from pathlib import Path
from urllib.parse import unquote, urlparse

CACHE_DIR_NAME = "huggingface-cache"


def cache_root(volume_path):
    """Where the platform mounts cached repositories, if a volume is mounted."""
    root = Path(volume_path) / CACHE_DIR_NAME / "hub"
    return root if root.is_dir() else None


def parse_repo_file(url):
    """Split a Hugging Face file URL into ``(repo_id, path_within_repo)``.

    Returns ``None`` for anything that is not a file link, including bare
    repository links, which name no file to go looking for.
    """
    parsed = urlparse(url)
    if parsed.hostname not in ("huggingface.co", "www.huggingface.co"):
        return None

    segments = [unquote(part) for part in parsed.path.split("/") if part]
    # <owner>/<name>/resolve/<revision>/<path...>
    if len(segments) < 5 or segments[2] not in ("resolve", "blob"):
        return None

    repo_id = f"{segments[0]}/{segments[1]}"
    return repo_id, "/".join(segments[4:])


def snapshot_dir(repo_id, root):
    """The cached snapshot for a repository, preferring the main branch."""
    model_root = Path(root) / f"models--{repo_id.replace('/', '--')}"
    snapshots = model_root / "snapshots"

    refs_main = model_root / "refs" / "main"
    if refs_main.is_file():
        try:
            candidate = snapshots / refs_main.read_text().strip()
        except OSError:
            candidate = None
        if candidate is not None and candidate.is_dir():
            return candidate

    if not snapshots.is_dir():
        return None
    # No main ref to trust, so pick deterministically rather than arbitrarily.
    versions = sorted(entry for entry in os.listdir(snapshots) if (snapshots / entry).is_dir())
    return snapshots / versions[0] if versions else None


def cached_file(url, root):
    """The cached copy of the file ``url`` points at, or ``None``."""
    if not root:
        return None

    parsed = parse_repo_file(url)
    if parsed is None:
        return None

    repo_id, path_within_repo = parsed
    snapshot = snapshot_dir(repo_id, root)
    if snapshot is None:
        return None

    candidate = snapshot / path_within_repo
    # The cache stores snapshots as links into blobs/, so follow before trusting.
    return candidate if candidate.is_file() else None
