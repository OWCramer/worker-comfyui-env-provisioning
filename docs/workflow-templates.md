# Workflow templates

The endpoint's API is `{"input": {"workflow": {...}}}`, which means sending a
request normally starts with writing a ComfyUI graph. A template removes that:
the image carries the graph, `COMFY_TEMPLATE` names which one, and the request
carries only what changes.

```
COMFY_TEMPLATE  = checkpoint
CHECKPOINT_URLS = https://huggingface.co/.../model.safetensors::model.safetensors
```

```json
{ "input": { "prompt": "a lighthouse at sunrise" } }
```

An empty `input` still renders, using the template's own defaults. A request
that sets `workflow` bypasses templates entirely, so nothing that worked before
behaves differently.

## Templates

| `COMFY_TEMPLATE` | Loads | Output |
| ---------------- | ----- | ------ |
| `checkpoint`      | one single-file checkpoint (SD 1.5, SDXL, fine-tunes) | image |
| `checkpoint-lora` | that, plus a LoRA                                     | image |
| `flux`            | a single-file FLUX checkpoint                         | image |
| `flux-lora`       | that, plus a LoRA                                     | image |
| `z-image`         | Z-Image Turbo diffusion model, Qwen3 encoder, its VAE   | image |
| `qwen-image`      | Qwen-Image diffusion model, Qwen2.5-VL encoder, its VAE | image |
| `krea2`           | Krea 2 Turbo diffusion model, Qwen3-VL encoder, Qwen image VAE | image |
| `wan-ti2v`        | Wan 2.2 TI2V diffusion model, umt5 encoder, Wan 2.2 VAE | video |
| `wan-t2v`         | Wan 2.1 diffusion model, umt5 encoder, Wan 2.1 VAE      | video |

Each accepts the parameters listed in its `defaults` block. Sending one the
template does not declare is an error naming the ones it does, rather than being
silently dropped.

Models are referenced by the fixed filenames the provisioning variables write
(`model.safetensors`, `lora.safetensors`, `text_encoder.safetensors`,
`vae.safetensors`), so a template never depends on what a repository called its
file. See [environment-provisioning.md](environment-provisioning.md).

## Deploy-time defaults

`COMFY_TEMPLATE_DEFAULTS` is a JSON object of per-model values the template
cannot know, applied between its defaults and the request:

```
COMFY_TEMPLATE_DEFAULTS = {"steps": 12, "trigger_words": "advt, "}
```

The LoRA templates prepend `trigger_words` to the prompt, because a LoRA trained
on a trigger renders nothing of itself without it.

## Adding one

Drop a `templates/<name>.json` in with `defaults` and `graph`. A string that is
exactly `{{name}}` is replaced by that parameter with its type intact, so
`"steps": "{{steps}}"` stays an integer; `{{name}}` inside a longer string is
interpolated as text. No code changes, and `tests/behavioral/
test_persona_template_sender.py` checks every shipped template builds from its
own defaults with no dangling node references.
