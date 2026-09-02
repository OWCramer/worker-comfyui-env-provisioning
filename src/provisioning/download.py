"""Download resolved models into the ComfyUI models tree.

Behavioral guarantees:

- A model Runpod has already cached on the host is linked into place instead of
  downloaded, which is the difference between a first request that waits on
  tens of gigabytes and one that starts immediately.
- Models land on the network volume (``<volume>/models/<type>/``) when one is
  mounted — matching ``extra_model_paths.yaml`` — so a fleet of workers pays
  each download once. Without a volume they land in ``<comfy_home>/models/``.
- Idempotent: a file that already exists is never re-downloaded.
- Concurrent-safe: a ``<file>.lock`` marker keeps two cold-starting workers
  from downloading the same file; the loser waits for the winner and then
  skips. Stale locks (from crashed workers) are broken after a timeout.
- Atomic: downloads stream to ``<file>.part`` and are renamed into place, so
  a crash never leaves a truncated file that looks complete.
"""

import os
import time
from dataclasses import dataclass
from pathlib import Path

from . import hf_cache
from .errors import DownloadError

DOWNLOAD_RETRIES = 3
RETRY_BACKOFF_SECONDS = 2.0
LOCK_POLL_SECONDS = 1.0

_TOKEN_HINTS = {
    "civitai": "set the CIVITAI_TOKEN environment variable (https://civitai.com/user/account)",
    "huggingface": "set the HF_TOKEN environment variable (https://huggingface.co/settings/tokens)",
    "direct": "verify the URL is publicly accessible",
}


@dataclass
class DownloadResult:
    filename: str
    directory: str
    path: str
    status: str  # "downloaded" | "cached" | "skipped"
    source: str
    url: str


def models_root(comfy_home, volume_path):
    """Where models should be written: the network volume when mounted."""
    volume = Path(volume_path)
    if volume.is_dir():
        return volume / "models", True
    return Path(comfy_home) / "models", False


def _acquire_lock(lock_path, final_path, *, sleep, lock_wait_seconds, lock_stale_seconds):
    """Try to own the download. Returns False if another worker finished it."""
    waited = 0.0
    while True:
        try:
            fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.write(fd, str(os.getpid()).encode())
            os.close(fd)
            return True
        except FileExistsError:
            if final_path.exists():
                return False
            try:
                age = time.time() - os.path.getmtime(lock_path)
            except OSError:
                continue  # lock vanished between checks — retry acquisition
            if age > lock_stale_seconds:
                try:
                    os.unlink(lock_path)
                except FileNotFoundError:
                    pass
                continue
            if waited >= lock_wait_seconds:
                raise DownloadError(
                    f"Timed out after {int(waited)}s waiting for another worker "
                    f"to finish downloading {final_path.name}. If no other worker "
                    f"is active, delete the stale lock file at {lock_path}."
                )
            sleep(LOCK_POLL_SECONDS)
            waited += LOCK_POLL_SECONDS


def _link_into_place(source, final_path):
    """Point final_path at an already-present file, atomically.

    A symlink rather than a copy: the cached file is often tens of gigabytes,
    and it may sit on a different filesystem from the models tree, which rules
    out a hard link.
    """
    link_path = str(final_path) + ".link"
    try:
        os.unlink(link_path)
    except FileNotFoundError:
        pass
    os.symlink(os.fspath(source), link_path)
    os.replace(link_path, final_path)


def _stream_to_file(response, part_path):
    with open(part_path, "wb") as fh:
        for chunk in response.iter_content(chunk_size=8 * 1024 * 1024):
            if chunk:
                fh.write(chunk)


def _fetch(resolved, spec, part_path, *, session, sleep):
    last_error = None
    for attempt in range(1, DOWNLOAD_RETRIES + 1):
        try:
            response = session.get(
                resolved.download_url, headers=resolved.headers, stream=True
            )
        except Exception as exc:  # connection errors, DNS, timeouts
            last_error = f"connection error: {exc}"
            sleep(RETRY_BACKOFF_SECONDS * attempt)
            continue

        status = response.status_code
        if status == 200:
            _stream_to_file(response, part_path)
            return
        if status in (401, 403):
            raise DownloadError(
                f"{spec.env_var}: access denied ({status}) downloading "
                f"{resolved.filename} from {spec.url} — "
                f"{_TOKEN_HINTS[resolved.source]}."
            )
        if 400 <= status < 500:
            raise DownloadError(
                f"{spec.env_var}: HTTP {status} downloading {resolved.filename} "
                f"from {spec.url} — check the URL."
            )
        last_error = f"HTTP {status}"
        sleep(RETRY_BACKOFF_SECONDS * attempt)

    raise DownloadError(
        f"{spec.env_var}: failed to download {resolved.filename} from "
        f"{spec.url} after {DOWNLOAD_RETRIES} attempts ({last_error})."
    )


def download_model(
    resolved,
    spec,
    *,
    root,
    session,
    cache_root=None,
    sleep=time.sleep,
    lock_wait_seconds=1800.0,
    lock_stale_seconds=3600.0,
):
    """Download one resolved model into ``<root>/<spec.directory>/``."""
    # Filenames can come from remote metadata (e.g. the Civitai API), so they
    # are untrusted even when the user's own config validated cleanly.
    name = resolved.filename
    if (
        not name
        or name in (".", "..")
        or "/" in name
        or "\\" in name
        or os.path.basename(name) != name
    ):
        raise DownloadError(
            f"{spec.env_var}: refusing unsafe filename {name!r} for {spec.url} — "
            "override it with '::<filename>'."
        )

    target_dir = Path(root) / spec.directory
    target_dir.mkdir(parents=True, exist_ok=True)
    final_path = target_dir / resolved.filename

    def result(status):
        return DownloadResult(
            filename=resolved.filename,
            directory=spec.directory,
            path=str(final_path),
            status=status,
            source=resolved.source,
            url=spec.url,
        )

    if final_path.exists():
        return result("skipped")

    # Before the lock, because linking is instant and the rename is atomic, so
    # two workers racing here cannot leave a half-written file between them.
    cached = hf_cache.cached_file(spec.url, cache_root)
    if cached is not None:
        _link_into_place(cached, final_path)
        return result("cached")

    lock_path = str(final_path) + ".lock"
    if not _acquire_lock(
        Path(lock_path),
        final_path,
        sleep=sleep,
        lock_wait_seconds=lock_wait_seconds,
        lock_stale_seconds=lock_stale_seconds,
    ):
        return result("skipped")

    part_path = str(final_path) + ".part"
    try:
        if final_path.exists():  # finished by someone else while we acquired
            return result("skipped")
        _fetch(resolved, spec, part_path, session=session, sleep=sleep)
        os.replace(part_path, final_path)
        return result("downloaded")
    finally:
        for cleanup in (part_path, lock_path):
            try:
                os.unlink(cleanup)
            except FileNotFoundError:
                pass
