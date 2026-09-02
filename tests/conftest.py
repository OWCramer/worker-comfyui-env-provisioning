"""Shared fixtures for the worker-comfyui test suite.

The behavioral tests simulate real user personas configuring a serverless
endpoint. Nothing touches the network or a real ComfyUI install: HTTP is
served by FakeSession (with a small Civitai API simulator on top) and
``comfy-node-install`` is simulated by FakeNodeRunner, which mimics the side
effects of a real install (node dir appears in custom_nodes/).
"""

import json
import os
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))  # handler.py lives at the repo root
sys.path.insert(0, str(REPO_ROOT / "src"))  # provisioning package lives in src/


# --------------------------------------------------------------------------
# Fake HTTP layer
# --------------------------------------------------------------------------


class FakeResponse:
    def __init__(self, status_code=200, json_data=None, content=b"", text=""):
        self.status_code = status_code
        self._json = json_data
        self._content = content
        self.text = text or (json.dumps(json_data) if json_data is not None else "")
        self.headers = {}

    def json(self):
        if self._json is None:
            raise ValueError("No JSON body")
        return self._json

    def iter_content(self, chunk_size=8192):
        for i in range(0, len(self._content), chunk_size):
            yield self._content[i : i + chunk_size]


class FakeSession:
    """URL → response map that records every request it serves.

    Responses may be a FakeResponse, a list of FakeResponses (consumed in
    order — for retry tests), or a callable(url, headers) -> FakeResponse.
    URLs are matched exactly first, then by prefix (longest prefix wins) so
    tests don't have to predict query-parameter ordering.
    """

    def __init__(self):
        self.routes = {}
        self.requests = []  # (url, headers) in call order

    def register(self, url, response):
        self.routes[url] = response

    def _lookup(self, url):
        if url in self.routes:
            return url
        candidates = [route for route in self.routes if url.startswith(route)]
        if candidates:
            return max(candidates, key=len)
        return None

    def get(self, url, headers=None, stream=False, **kwargs):
        self.requests.append((url, dict(headers or {})))
        route = self._lookup(url)
        if route is None:
            return FakeResponse(status_code=404, text=f"no fake route for {url}")
        response = self.routes[route]
        if isinstance(response, list):
            entry = response.pop(0) if len(response) > 1 else response[0]
            response = entry
        if callable(response):
            response = response(url, headers)
        return response

    def requested_urls(self):
        return [url for url, _ in self.requests]

    def download_count(self, url_prefix):
        return sum(1 for url in self.requested_urls() if url.startswith(url_prefix))


class CivitaiSimulator:
    """Registers realistic Civitai API + download routes on a FakeSession."""

    def __init__(self, session):
        self.session = session

    def add_model(
        self,
        model_id,
        *,
        model_type="Checkpoint",
        versions,
        requires_token=False,
    ):
        """versions: list of dicts {id, filename, content}."""

        def guard(payload):
            def handle(url, headers):
                authed = (headers or {}).get("Authorization") or "token=" in url
                if requires_token and not authed:
                    return FakeResponse(status_code=401, text="Unauthorized")
                return payload() if callable(payload) else payload

            return handle

        model_payload = FakeResponse(
            json_data={
                "id": model_id,
                "type": model_type,
                "modelVersions": [
                    {
                        "id": v["id"],
                        "files": [
                            {
                                "name": v["filename"],
                                "primary": True,
                                "downloadUrl": f"https://civitai.com/api/download/models/{v['id']}",
                            }
                        ],
                    }
                    for v in versions
                ],
            }
        )
        self.session.register(
            f"https://civitai.com/api/v1/models/{model_id}", guard(model_payload)
        )

        for v in versions:
            version_payload = FakeResponse(
                json_data={
                    "id": v["id"],
                    "model": {"type": model_type},
                    "files": [
                        {
                            "name": v["filename"],
                            "primary": True,
                            "downloadUrl": f"https://civitai.com/api/download/models/{v['id']}",
                        }
                    ],
                }
            )
            self.session.register(
                f"https://civitai.com/api/v1/model-versions/{v['id']}",
                guard(version_payload),
            )
            self.session.register(
                f"https://civitai.com/api/download/models/{v['id']}",
                guard(FakeResponse(content=v.get("content", b"model-bytes"))),
            )


# --------------------------------------------------------------------------
# Fake node installer
# --------------------------------------------------------------------------


class FakeProc:
    def __init__(self, returncode=0, stdout="", stderr=""):
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr


class FakeNodeRunner:
    """Simulates subprocess.run for comfy-node-install and pip.

    A comfy-node-install call creates ``custom_nodes/<DirName>/`` (plus a
    requirements.txt when configured), matching the real side effect. Node
    ids map to differently-cased dir names, like the real registry.
    """

    def __init__(self, custom_nodes_dir):
        self.custom_nodes_dir = Path(custom_nodes_dir)
        self.calls = []  # raw argv lists
        self.failing_nodes = set()
        self.requirements = {}  # node name -> requirements.txt content

    def node_dir_name(self, name):
        return "".join(part.capitalize() for part in name.split("-"))

    def __call__(self, argv, capture_output=False, text=False, **kwargs):
        self.calls.append(list(argv))
        if argv[0] == "comfy-node-install":
            spec = argv[1]
            name = spec.split("@")[0]
            if name in self.failing_nodes:
                return FakeProc(
                    returncode=1,
                    stderr=f"Comfy node installation failed for: {name}",
                )
            node_dir = self.custom_nodes_dir / self.node_dir_name(name)
            node_dir.mkdir(parents=True, exist_ok=True)
            (node_dir / "__init__.py").write_text("NODE_CLASS_MAPPINGS = {}\n")
            if name in self.requirements:
                (node_dir / "requirements.txt").write_text(self.requirements[name])
            return FakeProc()
        if argv[:4] == ["python", "-m", "pip", "install"]:
            if "--target" in argv:
                target = Path(argv[argv.index("--target") + 1])
                target.mkdir(parents=True, exist_ok=True)
                (target / "fake_dep").mkdir(exist_ok=True)
            return FakeProc()
        return FakeProc(returncode=1, stderr=f"unexpected command: {argv}")

    def install_calls(self):
        return [c[1] for c in self.calls if c[0] == "comfy-node-install"]


# --------------------------------------------------------------------------
# Worker environment
# --------------------------------------------------------------------------


class WorkerEnv:
    """A sandboxed stand-in for the worker container filesystem."""

    def __init__(self, tmp_path, with_volume):
        self.comfy_home = tmp_path / "comfyui"
        (self.comfy_home / "models").mkdir(parents=True)
        (self.comfy_home / "custom_nodes").mkdir(parents=True)
        self.volume_path = tmp_path / "runpod-volume"
        if with_volume:
            self.volume_path.mkdir()
        self.env_file = tmp_path / "provision_env.sh"
        self.manifest_path = tmp_path / "provision_manifest.json"
        self.session = FakeSession()
        self.civitai = CivitaiSimulator(self.session)
        self.node_runner = FakeNodeRunner(self.comfy_home / "custom_nodes")
        self.environ = {}

    @property
    def models_root(self):
        return (
            self.volume_path / "models"
            if self.volume_path.is_dir()
            else self.comfy_home / "models"
        )

    def bake_model(self, directory, filename, content=b"baked"):
        """Simulate a model already present in the image."""
        path = self.comfy_home / "models" / directory / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
        return path

    def cache_model(self, repo_id, path_within_repo, content=b"cached", revision="c0ffee"):
        """Simulate a repository Runpod has already cached on the host."""
        root = (
            self.volume_path
            / "huggingface-cache"
            / "hub"
            / f"models--{repo_id.replace('/', '--')}"
        )
        target = root / "snapshots" / revision / path_within_repo
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(content)
        (root / "refs").mkdir(parents=True, exist_ok=True)
        (root / "refs" / "main").write_text(revision)
        return target

    def bake_node(self, dir_name):
        """Simulate a custom node already baked into the image."""
        path = self.comfy_home / "custom_nodes" / dir_name
        path.mkdir(parents=True, exist_ok=True)
        return path

    def provision(self, **overrides):
        from provisioning import provision

        kwargs = dict(
            environ=self.environ,
            session=self.session,
            comfy_home=str(self.comfy_home),
            volume_path=str(self.volume_path),
            env_file=str(self.env_file),
            manifest_path=str(self.manifest_path),
            node_runner=self.node_runner,
            sleep=lambda seconds: None,
            lock_wait_seconds=5.0,
            lock_stale_seconds=60.0,
        )
        kwargs.update(overrides)
        return provision(**kwargs)

    def model_path(self, directory, filename):
        return self.models_root / directory / filename

    def manifest_json(self):
        return json.loads(self.manifest_path.read_text())


@pytest.fixture
def worker(tmp_path):
    """A worker with a network volume attached (the recommended setup)."""
    return WorkerEnv(tmp_path, with_volume=True)


@pytest.fixture
def worker_no_volume(tmp_path):
    """A worker without a network volume."""
    return WorkerEnv(tmp_path, with_volume=False)
