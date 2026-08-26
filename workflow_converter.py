"""Convert ComfyUI UI-format workflow exports into API format.

Users' natural artifact is the UI export (Workflow -> Save, or the `workflow`
chunk embedded in generated PNGs): a graph of ``nodes`` and ``links`` with
positional ``widgets_values``. The worker executes API format: a mapping of
node id -> {class_type, inputs}. ComfyUI only performs this conversion in its
frontend, so the worker does it server-side using the live ComfyUI's
``/object_info`` schema, which describes every node's inputs in order.

Handled UI-graph realities:
- ``widgets_values`` are positional; the schema says which inputs are widgets.
- Seed-style inputs carry a phantom ``control_after_generate`` widget value
  that must be skipped (flagged in the schema as ``control_after_generate``).
- ``Note``/``MarkdownNote`` nodes are annotations — dropped.
- ``Reroute`` nodes are wiring sugar — links are followed through them.
- Muted/bypassed nodes (mode 2 or 4) are dropped; links through them break
  with a clear error rather than silently producing a wrong image.
"""

class WorkflowConversionError(Exception):
    """The UI-format workflow could not be converted faithfully."""


# Nodes that exist only for humans reading the graph.
ANNOTATION_NODES = {"Note", "MarkdownNote"}
# Nodes that only forward a connection.
PASSTHROUGH_NODES = {"Reroute"}
# Node modes: 0/1 = active, 2 = muted (produces nothing), 4 = bypassed
# (acts as a passthrough: forwards its matching-type input to its output).
MODE_MUTED = 2
MODE_BYPASSED = 4

# Input types that come from a link, never from a widget value.
_CONNECTION_ONLY_TYPES = {
    "MODEL", "CLIP", "VAE", "CONDITIONING", "LATENT", "IMAGE", "MASK",
    "CONTROL_NET", "STYLE_MODEL", "CLIP_VISION", "CLIP_VISION_OUTPUT",
    "GLIGEN", "UPSCALE_MODEL", "SAMPLER", "SIGMAS", "GUIDER", "NOISE",
    "AUDIO", "VIDEO", "WEBCAM", "PHOTOMAKER",
}
_WIDGET_TYPES = {"INT", "FLOAT", "STRING", "BOOLEAN"}


def is_ui_format(workflow):
    """A UI export has a top-level ``nodes`` list; API format is a dict of
    node-id -> {class_type, ...}."""
    return isinstance(workflow, dict) and isinstance(workflow.get("nodes"), list)


def _input_is_widget(input_type, config):
    """Decide whether an input consumes a widget value.

    Combo inputs are lists of options; primitives are INT/FLOAT/STRING/BOOLEAN.
    Anything typed as a node-output connection is link-only.
    """
    if isinstance(input_type, list):
        return True  # combo box
    if input_type in _WIDGET_TYPES:
        return True
    if input_type in _CONNECTION_ONLY_TYPES:
        return False
    # Unknown custom type: treat as connection (safe default — widget
    # misalignment corrupts every later input, a missing widget only one).
    return False


def _iter_schema_inputs(node_schema):
    """Yield (name, type, config) over required then optional inputs, in order."""
    inputs = node_schema.get("input", {})
    for section in ("required", "optional"):
        for name, spec in (inputs.get(section) or {}).items():
            input_type = spec[0] if isinstance(spec, (list, tuple)) and spec else spec
            config = (
                spec[1]
                if isinstance(spec, (list, tuple)) and len(spec) > 1 and isinstance(spec[1], dict)
                else {}
            )
            yield name, input_type, config


def _resolve_source(node_id, output_index, nodes_by_id, links_by_id):
    """Follow Reroute chains and bypassed nodes back to a real producer.

    Bypassed nodes (mode 4) forward their input of the matching type to the
    requested output — ComfyUI's frontend does the same when executing.
    """
    seen = set()
    while True:
        node = nodes_by_id.get(node_id)
        if node is None:
            raise WorkflowConversionError(
                f"Workflow references missing node {node_id}."
            )
        is_reroute = node.get("type") in PASSTHROUGH_NODES
        is_bypassed = node.get("mode") == MODE_BYPASSED
        if not is_reroute and not is_bypassed:
            return node_id, output_index
        if node_id in seen:
            raise WorkflowConversionError(
                f"Passthrough cycle detected at node {node_id}."
            )
        seen.add(node_id)

        connected = [i for i in (node.get("inputs") or []) if i.get("link") is not None]
        if is_reroute:
            upstream = connected[0] if connected else None
        else:
            # Bypass: forward the input whose type matches the requested output.
            outputs = node.get("outputs") or []
            wanted_type = (
                outputs[output_index].get("type")
                if output_index < len(outputs)
                else None
            )
            upstream = next(
                (i for i in connected if i.get("type") == wanted_type), None
            )
        if upstream is None:
            raise WorkflowConversionError(
                f"Node {node_id} ({node.get('type')}) is bypassed/rerouted but has "
                "no matching incoming connection to forward. Unbypass it or remove "
                "the downstream connection before exporting."
            )
        link = links_by_id.get(upstream["link"])
        if link is None:
            raise WorkflowConversionError(
                f"Node {node_id} references missing link {upstream['link']}."
            )
        node_id, output_index = str(link[1]), link[2]


def convert_ui_workflow(ui_workflow, object_info):
    """Convert a UI-format workflow into API format.

    Args:
        ui_workflow (dict): The UI export ({"nodes": [...], "links": [...]}).
        object_info (dict): ComfyUI /object_info schema (class_type -> schema).

    Returns:
        dict: API-format workflow (node id -> {class_type, inputs}).

    Raises:
        WorkflowConversionError: with an actionable message when the graph
        can't be converted faithfully.
    """
    nodes = ui_workflow.get("nodes") or []
    # links: [link_id, src_node, src_slot, dst_node, dst_slot, type]
    links_by_id = {l[0]: l for l in (ui_workflow.get("links") or []) if l}
    nodes_by_id = {str(n.get("id")): n for n in nodes}

    muted = {str(n.get("id")) for n in nodes if n.get("mode") == MODE_MUTED}
    bypassed = {str(n.get("id")) for n in nodes if n.get("mode") == MODE_BYPASSED}

    api_workflow = {}
    for node in nodes:
        node_id = str(node.get("id"))
        class_type = node.get("type")

        if class_type in ANNOTATION_NODES or class_type in PASSTHROUGH_NODES:
            continue
        if node_id in muted or node_id in bypassed:
            continue

        schema = object_info.get(class_type)
        if schema is None:
            raise WorkflowConversionError(
                f"Unknown node type '{class_type}' (node {node_id}). "
                "If this is a custom node, install it via CUSTOM_NODES or "
                "bake it into the image."
            )

        # Connected inputs by name, following reroutes to real producers.
        linked = {}
        for inp in node.get("inputs") or []:
            link_id = inp.get("link")
            if link_id is None:
                continue
            link = links_by_id.get(link_id)
            if link is None:
                raise WorkflowConversionError(
                    f"Node {node_id} input '{inp.get('name')}' references "
                    f"missing link {link_id}."
                )
            src_id, src_slot = _resolve_source(str(link[1]), link[2], nodes_by_id, links_by_id)
            if src_id in muted:
                raise WorkflowConversionError(
                    f"Node {node_id} input '{inp.get('name')}' is connected to "
                    f"muted node {src_id}. Unmute it or remove the connection "
                    "before exporting."
                )
            linked[inp.get("name")] = [src_id, src_slot]

        # Positional widget values mapped through the schema, in order.
        widget_values = list(node.get("widgets_values") or [])
        api_inputs = {}
        cursor = 0
        for name, input_type, config in _iter_schema_inputs(schema):
            if name in linked:
                api_inputs[name] = linked[name]
                # A linked widget-type input still occupies widget slots
                # (its value plus any control widget).
                if _input_is_widget(input_type, config):
                    cursor += 1
                    if config.get("control_after_generate"):
                        cursor += 1
                continue
            if not _input_is_widget(input_type, config):
                continue
            if cursor >= len(widget_values):
                if "default" in config:
                    api_inputs[name] = config["default"]
                    continue
                raise WorkflowConversionError(
                    f"Node {node_id} ({class_type}) is missing a value for "
                    f"input '{name}'."
                )
            api_inputs[name] = widget_values[cursor]
            cursor += 1
            # Seed-style inputs export a phantom 'control_after_generate'
            # value ("randomize"/"fixed"/...) right after the real value.
            if config.get("control_after_generate"):
                cursor += 1

        api_workflow[node_id] = {"class_type": class_type, "inputs": api_inputs}

    if not api_workflow:
        raise WorkflowConversionError(
            "Workflow contains no executable nodes after conversion."
        )
    return api_workflow
