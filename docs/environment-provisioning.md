# Custom models & nodes via environment variables

Deploy any custom model — a checkpoint from Civitai, a LoRA from Hugging Face,
your own hosted file — **without building a Docker image**. Set environment
variables on your endpoint and the worker downloads everything at startup.

## Quick start

Deploy the `base` image and set:

```
CHECKPOINT_URLS = https://civitai.com/models/4384/dreamshaper
CIVITAI_TOKEN   = <your Civitai API token>
```

That's it. At startup the worker resolves the model page through the Civitai
API, downloads the newest version into `models/checkpoints/`, and logs the
exact filename to reference in your workflow:

```
worker-comfyui (provisioning): Models ready — reference them in workflows by these filenames:
worker-comfyui (provisioning):   - dreamshaper_8.safetensors  (models/checkpoints, downloaded)
```

## Model environment variables

Each variable maps to one ComfyUI model directory. Entries are separated by
commas (or newlines):

| Variable               | Target directory          |
| ---------------------- | ------------------------- |
| `CHECKPOINT_URLS`      | `models/checkpoints`      |
| `LORA_URLS`            | `models/loras`            |
| `VAE_URLS`             | `models/vae`              |
| `CONTROLNET_URLS`      | `models/controlnet`       |
| `UPSCALE_MODEL_URLS`   | `models/upscale_models`   |
| `EMBEDDING_URLS`       | `models/embeddings`       |
| `CLIP_URLS`            | `models/clip`             |
| `CLIP_VISION_URLS`     | `models/clip_vision`      |
| `DIFFUSION_MODEL_URLS` | `models/diffusion_models` |
| `TEXT_ENCODER_URLS`    | `models/text_encoders`    |
| `UNET_URLS`            | `models/unet`             |

### Supported URL types

- **Civitai model pages** — `https://civitai.com/models/4384/dreamshaper`.
  The newest version is used; pick a specific one with
  `?modelVersionId=<id>` (the URL your browser shows when you select a
  version). The filename comes from the Civitai API.
- **Civitai download links** — `https://civitai.com/api/download/models/128713`.
- **Hugging Face file links** — `https://huggingface.co/<repo>/resolve/main/<file>`.
- **Any direct URL** whose path ends in a filename.

### Explicit filenames

Workflows reference models by filename. To control it, append `::<filename>`:

```
LORA_URLS = https://civitai.com/api/download/models/87153::add_detail.safetensors
```

### Authentication

| Variable        | Used for                                                                    |
| --------------- | --------------------------------------------------------------------------- |
| `CIVITAI_TOKEN` | Most Civitai downloads ([create one here](https://civitai.com/user/account)) |
| `HF_TOKEN`      | Gated/private Hugging Face models                                            |

## Custom nodes

Install Comfy Registry nodes at startup — no Dockerfile:

```
CUSTOM_NODES = comfyui-kjnodes@1.1.2, comfyui-ic-light@1.0.5
```

Find ids and versions at [registry.comfy.org](https://registry.comfy.org/).
**Pin versions in production** — unpinned nodes install the latest release,
so workers scaling up later could get different code (the worker logs a
warning when a node is unpinned).

Nodes that need system packages (apt) or conflicting torch versions still
require the [Dockerfile approach](customization.md).

## Attach a network volume (strongly recommended)

Without a volume, every cold-starting worker re-downloads every model. With a
[network volume](https://docs.runpod.io/storage/network-volumes) attached:

- models are stored on the volume (`/runpod-volume/models/...`) — the fleet
  pays each download **once**
- custom node installs are cached on the volume, so later cold starts skip
  the install entirely
- concurrent cold starts coordinate through lock files, so two workers never
  download the same file twice

## Behavior details

- Provisioning is **idempotent**: files that already exist are never
  re-downloaded.
- Provisioning **fails fast**: if any model or node can't be fetched, the
  worker exits with an actionable error instead of serving a broken endpoint.
- A machine-readable inventory is written to `/tmp/provision_manifest.json`.
- Kill switch: set `RUNTIME_PROVISIONING=false` to disable all of the above.
