"""Build a ComfyUI graph from a named template plus a few parameters.

The endpoint's own API is `{"input": {"workflow": {...}}}`, which means anyone
deploying a model has to hand-write a graph before they can send a request. A
template removes that: the image carries the graph, `COMFY_TEMPLATE` names which
one, and the request carries only what changes.

    {"input": {"prompt": "a lighthouse at sunrise"}}

Templates are data, not code (`templates/*.json`), so supporting a new model
family is a file rather than a release of whatever is generating requests.

Substitution is deliberately dull: a string that is exactly `{{name}}` becomes
that parameter's value with its type intact, so `"steps": "{{steps}}"` yields an
int. A `{{name}}` inside a longer string is interpolated as text.
"""

import json
import os
import re
from pathlib import Path

def _template_dir():
    """`/templates` in the image, `templates/` in a checkout."""
    beside = Path(__file__).resolve().parent / "templates"
    return beside if beside.is_dir() else Path(__file__).resolve().parent.parent / "templates"


TEMPLATE_DIR = _template_dir()

TEMPLATE_ENV_VAR = "COMFY_TEMPLATE"
DEFAULTS_ENV_VAR = "COMFY_TEMPLATE_DEFAULTS"

# Where startup provisioning records the template it detected for COMFY_MODEL.
DETECTED_TEMPLATE_FILE = "/tmp/provision_template"

PLACEHOLDER = re.compile(r"\{\{(\w+)\}\}")


class TemplateError(Exception):
    """A template could not be found, read, or filled in."""


def available(template_dir=TEMPLATE_DIR):
    return sorted(path.stem for path in Path(template_dir).glob("*.json"))


def selected_template(environ=None, detected_file=DETECTED_TEMPLATE_FILE):
    """The template to run: named by the deploy, else the one detected at startup."""
    environ = os.environ if environ is None else environ
    name = (environ.get(TEMPLATE_ENV_VAR) or "").strip()
    if name:
        return name

    try:
        detected = Path(detected_file).read_text().strip()
    except OSError:
        return None
    return detected or None


def load(name, template_dir=TEMPLATE_DIR):
    path = Path(template_dir) / f"{name}.json"
    if not path.is_file():
        raise TemplateError(
            f"Unknown {TEMPLATE_ENV_VAR} {name!r}. Available: {', '.join(available(template_dir))}."
        )
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError) as exc:
        raise TemplateError(f"Template {name!r} could not be read: {exc}") from exc


def _env_defaults(environ):
    """Per-model overrides the deploy baked in, e.g. a LoRA's trigger words."""
    raw = (environ.get(DEFAULTS_ENV_VAR) or "").strip()
    if not raw:
        return {}
    try:
        parsed = json.loads(raw)
    except ValueError as exc:
        raise TemplateError(f"{DEFAULTS_ENV_VAR} is not valid JSON: {exc}") from exc
    if not isinstance(parsed, dict):
        raise TemplateError(f"{DEFAULTS_ENV_VAR} must be a JSON object.")
    return parsed


def _fill(node, params, used):
    if isinstance(node, dict):
        return {key: _fill(value, params, used) for key, value in node.items()}
    if isinstance(node, list):
        return [_fill(item, params, used) for item in node]
    if not isinstance(node, str):
        return node

    whole = PLACEHOLDER.fullmatch(node)
    if whole:
        name = whole.group(1)
        used.add(name)
        if name not in params:
            raise TemplateError(f"Template needs a value for {name!r}.")
        return params[name]

    def interpolate(match):
        name = match.group(1)
        used.add(name)
        if name not in params:
            raise TemplateError(f"Template needs a value for {name!r}.")
        return str(params[name])

    return PLACEHOLDER.sub(interpolate, node)


def build(job_input, environ=None, template_dir=TEMPLATE_DIR, detected_file=DETECTED_TEMPLATE_FILE):
    """Expand the endpoint's template using the request's parameters.

    Precedence runs template defaults, then the deploy's overrides, then the
    request — so a caller sending nothing still gets a working graph, and a
    caller sending `prompt` overrides only that.
    """
    environ = os.environ if environ is None else environ

    name = selected_template(environ, detected_file)
    if name is None:
        return None

    template = load(name, template_dir)
    graph = template.get("graph")
    if not isinstance(graph, dict) or not graph:
        raise TemplateError(f"Template {name!r} has no graph.")

    params = dict(template.get("defaults") or {})
    params.update(_env_defaults(environ))
    params.update({k: v for k, v in (job_input or {}).items() if k != "workflow"})

    used = set()
    filled = _fill(graph, params, used)

    unknown = sorted(set(params) - used - {"images"})
    if unknown:
        raise TemplateError(
            f"Template {name!r} takes no parameter(s) {', '.join(unknown)}. "
            f"It accepts: {', '.join(sorted(used))}."
        )

    return filled
