# Behavioral test suite: user personas

This suite is the behavioral contract for runtime provisioning (custom models
and custom nodes via environment variables). Each file is a **persona** — a
real kind of user deploying this worker — and its tests describe what that
user must be able to do. If you change provisioning behavior, change the
persona's tests first, then make the implementation match.

Everything runs hermetically: HTTP is faked (including a small Civitai API
simulator) and `comfy-node-install` is simulated with its real side effects.
No network, no GPU, no ComfyUI install needed. See `tests/conftest.py`.

| File                             | Persona | They want                                                                 |
| -------------------------------- | ------- | ------------------------------------------------------------------------- |
| `test_persona_baseline.py`       | Dana    | A packaged image with zero config — provisioning must be invisible (regression guard + kill switch). |
| `test_persona_model_runner.py`   | Priya   | One Civitai checkpoint, pasted as a URL, no Docker knowledge. Resolution, filenames, tokens, placement. |
| `test_persona_lora_stacker.py`   | Marco   | A checkpoint + LoRA stack swapped weekly on a network volume. Idempotency, fleet caching, concurrent cold starts, flaky sources. |
| `test_persona_node_tinkerer.py`  | Sofia   | A handful of pinned registry custom nodes without a Dockerfile. Installs, volume cache, cache invalidation, failure guidance. |
| `test_persona_power_user.py`     | Ken     | A full production Flux pipeline: many models across all types + several nodes. Ordering, manifest inventory, determinism, fail-fast, scale. |
| `test_persona_hostile_input.py`  | Mallory | To break it: typos, traversal filenames, shell metacharacters, garbage APIs. Every failure is a clear config error; nothing escapes the sandbox. |
| `test_persona_video_creator.py`  | Vera    | Video workflows (Wan, Hunyuan, LTX, SVD). Every video-save node reporting style returns files under `output.videos`; image behavior untouched. |
| `test_persona_ui_exporter.py`    | Uma     | Sends the raw ComfyUI UI export (Workflow -> Save). Server-side conversion to API format: widget mapping, seed control skips, notes, reroutes — or actionable errors. |

Run with:

```bash
pytest tests/behavioral -v
```
