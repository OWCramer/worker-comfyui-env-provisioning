"""Persona: Uma — exports her workflow from the ComfyUI menu, as-is.

Uma clicks Workflow -> Save (or drags a PNG out of ComfyUI) and sends that
JSON to the endpoint. She has never heard of "API format". The worker must
convert her UI-format graph faithfully: positional widgets_values mapped by
the node schema, the phantom control_after_generate value after seeds
skipped, notes dropped, reroutes followed — or fail with advice she can act
on, never a silently wrong image.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from workflow_converter import (
    WorkflowConversionError,
    convert_ui_workflow,
    is_ui_format,
)

# Minimal /object_info schema for the nodes Uma's graph uses.
OBJECT_INFO = {
    "CheckpointLoaderSimple": {
        "input": {"required": {"ckpt_name": [["v1-5-pruned-emaonly-fp16.safetensors"]]}}
    },
    "CLIPTextEncode": {
        "input": {
            "required": {
                "text": ["STRING", {"multiline": True}],
                "clip": ["CLIP"],
            }
        }
    },
    "EmptyLatentImage": {
        "input": {
            "required": {
                "width": ["INT", {"default": 512}],
                "height": ["INT", {"default": 512}],
                "batch_size": ["INT", {"default": 1}],
            }
        }
    },
    "KSampler": {
        "input": {
            "required": {
                "model": ["MODEL"],
                "seed": ["INT", {"default": 0, "control_after_generate": True}],
                "steps": ["INT", {"default": 20}],
                "cfg": ["FLOAT", {"default": 8.0}],
                "sampler_name": [["euler", "euler_ancestral"]],
                "scheduler": [["normal", "karras"]],
                "positive": ["CONDITIONING"],
                "negative": ["CONDITIONING"],
                "latent_image": ["LATENT"],
                "denoise": ["FLOAT", {"default": 1.0}],
            }
        }
    },
    "VAEDecode": {
        "input": {"required": {"samples": ["LATENT"], "vae": ["VAE"]}}
    },
    "SaveImage": {
        "input": {
            "required": {
                "images": ["IMAGE"],
                "filename_prefix": ["STRING", {"default": "ComfyUI"}],
            }
        }
    },
}


def node(nid, ntype, inputs=None, outputs=None, widgets=None, mode=0):
    return {
        "id": nid,
        "type": ntype,
        "mode": mode,
        "inputs": inputs or [],
        "outputs": outputs or [],
        "widgets_values": widgets or [],
    }


def link(lid, src, src_slot, dst, dst_slot, ltype="*"):
    return [lid, src, src_slot, dst, dst_slot, ltype]


@pytest.fixture
def uma_export():
    """A faithful UI export of the standard SD1.5 txt2img graph, including
    the phantom control_after_generate widget value after the seed and a
    Note node with her todo list."""
    return {
        "nodes": [
            node(4, "CheckpointLoaderSimple",
                 widgets=["v1-5-pruned-emaonly-fp16.safetensors"]),
            node(6, "CLIPTextEncode",
                 inputs=[{"name": "clip", "type": "CLIP", "link": 3}],
                 widgets=["a cozy cabin in a snowy forest"]),
            node(7, "CLIPTextEncode",
                 inputs=[{"name": "clip", "type": "CLIP", "link": 5}],
                 widgets=["blurry, low quality"]),
            node(5, "EmptyLatentImage", widgets=[512, 512, 1]),
            node(3, "KSampler",
                 inputs=[
                     {"name": "model", "type": "MODEL", "link": 1},
                     {"name": "positive", "type": "CONDITIONING", "link": 4},
                     {"name": "negative", "type": "CONDITIONING", "link": 6},
                     {"name": "latent_image", "type": "LATENT", "link": 2},
                 ],
                 # [seed, control_after_generate, steps, cfg, sampler, scheduler, denoise]
                 widgets=[42, "randomize", 20, 7.5, "euler", "normal", 1.0]),
            node(8, "VAEDecode",
                 inputs=[
                     {"name": "samples", "type": "LATENT", "link": 7},
                     {"name": "vae", "type": "VAE", "link": 8},
                 ]),
            node(9, "SaveImage",
                 inputs=[{"name": "images", "type": "IMAGE", "link": 9}],
                 widgets=["ComfyUI"]),
            node(20, "Note", widgets=["try 30 steps next time!!"]),
        ],
        "links": [
            link(1, 4, 0, 3, 0, "MODEL"),
            link(3, 4, 1, 6, 0, "CLIP"),
            link(5, 4, 1, 7, 0, "CLIP"),
            link(4, 6, 0, 3, 1, "CONDITIONING"),
            link(6, 7, 0, 3, 2, "CONDITIONING"),
            link(2, 5, 0, 3, 3, "LATENT"),
            link(7, 3, 0, 8, 0, "LATENT"),
            link(8, 4, 2, 8, 1, "VAE"),
            link(9, 8, 0, 9, 0, "IMAGE"),
        ],
    }


class TestFormatDetection:
    def test_ui_export_is_detected(self, uma_export):
        assert is_ui_format(uma_export)

    def test_api_format_is_not_detected(self):
        api = {"3": {"class_type": "KSampler", "inputs": {}}}
        assert not is_ui_format(api)

    def test_api_format_with_numeric_ids_is_not_detected(self):
        assert not is_ui_format({"nodes": {"3": {}}})  # dict, not list


class TestFaithfulConversion:
    def test_full_graph_converts(self, uma_export):
        api = convert_ui_workflow(uma_export, OBJECT_INFO)
        assert set(api.keys()) == {"4", "6", "7", "5", "3", "8", "9"}

    def test_widget_values_map_to_named_inputs(self, uma_export):
        api = convert_ui_workflow(uma_export, OBJECT_INFO)
        assert api["4"]["inputs"]["ckpt_name"] == "v1-5-pruned-emaonly-fp16.safetensors"
        assert api["6"]["inputs"]["text"] == "a cozy cabin in a snowy forest"
        assert api["5"]["inputs"] == {"width": 512, "height": 512, "batch_size": 1}

    def test_control_after_generate_phantom_value_is_skipped(self, uma_export):
        """The 'randomize' after the seed must not shift later widgets."""
        api = convert_ui_workflow(uma_export, OBJECT_INFO)
        ks = api["3"]["inputs"]
        assert ks["seed"] == 42
        assert ks["steps"] == 20
        assert ks["cfg"] == 7.5
        assert ks["sampler_name"] == "euler"
        assert ks["scheduler"] == "normal"
        assert ks["denoise"] == 1.0
        assert "control_after_generate" not in ks

    def test_connections_become_node_references(self, uma_export):
        api = convert_ui_workflow(uma_export, OBJECT_INFO)
        assert api["3"]["inputs"]["model"] == ["4", 0]
        assert api["3"]["inputs"]["positive"] == ["6", 0]
        assert api["8"]["inputs"]["vae"] == ["4", 2]

    def test_note_nodes_are_dropped(self, uma_export):
        api = convert_ui_workflow(uma_export, OBJECT_INFO)
        assert "20" not in api


class TestReroutes:
    def test_links_are_followed_through_reroute_chains(self, uma_export):
        # Uma routes the model connection through two tidy reroutes.
        uma_export["nodes"].append(
            node(30, "Reroute",
                 inputs=[{"name": "", "type": "*", "link": 100}])
        )
        uma_export["nodes"].append(
            node(31, "Reroute",
                 inputs=[{"name": "", "type": "*", "link": 101}])
        )
        uma_export["links"].append(link(100, 4, 0, 30, 0))   # ckpt -> reroute A
        uma_export["links"].append(link(101, 30, 0, 31, 0))  # A -> B
        uma_export["links"].append(link(102, 31, 0, 3, 0))   # B -> KSampler.model
        # repoint KSampler.model at the reroute chain
        for n in uma_export["nodes"]:
            if n["id"] == 3:
                n["inputs"][0]["link"] = 102

        api = convert_ui_workflow(uma_export, OBJECT_INFO)
        assert api["3"]["inputs"]["model"] == ["4", 0]
        assert "30" not in api and "31" not in api


class TestCustomNodeSeeds:
    """Custom nodes declare seed as a plain INT (no control_after_generate
    flag), but the frontend still appends the control widget by input-name
    convention. Found live: UmeAiRT_VideoSettings fed 'randomize' into
    frame_rate, shifting every later widget by one."""

    def test_unflagged_seed_still_skips_the_control_value(self):
        schema = {
            "UmeAiRT_VideoSettings": {
                "input": {
                    "required": {
                        "width": ["INT", {"default": 512}],
                        "height": ["INT", {"default": 512}],
                        "seconds": ["INT", {"default": 3}],
                        "steps": ["INT", {"default": 20}],
                        "cfg": ["FLOAT", {"default": 6.0}],
                        "shift": ["FLOAT", {"default": 6.0}],
                        "sampler": [["euler"]],
                        "scheduler": [["simple"]],
                        "seed": ["INT", {"default": 0}],  # NO flag
                        "frame_rate": ["INT", {"default": 16}],
                    }
                }
            }
        }
        ui = {
            "nodes": [node(1, "UmeAiRT_VideoSettings",
                           widgets=[880, 1120, 3, 20, 6, 6, "euler", "simple",
                                    924303502919000, "randomize", 16])],
            "links": [],
        }
        api = convert_ui_workflow(ui, schema)
        ins = api["1"]["inputs"]
        assert ins["seed"] == 924303502919000
        assert ins["frame_rate"] == 16  # NOT 'randomize'

    def test_seed_without_exported_control_value_does_not_overskip(self):
        """Some exports omit the control value — peek before skipping."""
        schema = {
            "SeedOnly": {
                "input": {
                    "required": {
                        "seed": ["INT", {"default": 0}],
                        "steps": ["INT", {"default": 20}],
                    }
                }
            }
        }
        ui = {"nodes": [node(1, "SeedOnly", widgets=[42, 30])], "links": []}
        api = convert_ui_workflow(ui, schema)
        assert api["1"]["inputs"] == {"seed": 42, "steps": 30}


class TestBypassedNodes:
    """Bypass (mode 4) is a passthrough — ComfyUI forwards the node's
    matching-type input. Community workflows toggle whole groups off this
    way (Fast Groups Bypasser), so getting this wrong false-fails most
    real workflows."""

    def test_bypassed_node_is_forwarded_not_fatal(self, uma_export):
        # Uma bypasses a LatentUpscaleBy between EmptyLatentImage and KSampler.
        schema = dict(OBJECT_INFO)
        schema["LatentUpscaleBy"] = {
            "input": {
                "required": {
                    "samples": ["LATENT"],
                    "upscale_method": [["nearest-exact", "bilinear"]],
                    "scale_by": ["FLOAT", {"default": 1.5}],
                }
            }
        }
        uma_export["nodes"].append(
            node(40, "LatentUpscaleBy", mode=4,
                 inputs=[{"name": "samples", "type": "LATENT", "link": 200}],
                 outputs=[{"name": "LATENT", "type": "LATENT"}],
                 widgets=["nearest-exact", 1.5])
        )
        uma_export["links"].append(link(200, 5, 0, 40, 0, "LATENT"))
        uma_export["links"].append(link(201, 40, 0, 3, 3, "LATENT"))
        for n in uma_export["nodes"]:
            if n["id"] == 3:
                n["inputs"][3]["link"] = 201

        api = convert_ui_workflow(uma_export, schema)
        # KSampler's latent comes straight from EmptyLatentImage; the
        # bypassed upscaler is absent from the executable graph.
        assert api["3"]["inputs"]["latent_image"] == ["5", 0]
        assert "40" not in api

    def test_muted_node_is_still_a_hard_error(self, uma_export):
        uma_export["nodes"][0]["mode"] = 2  # mute the checkpoint loader
        with pytest.raises(WorkflowConversionError) as excinfo:
            convert_ui_workflow(uma_export, OBJECT_INFO)
        assert "muted" in str(excinfo.value).lower()


class TestBypassWithNothingToForward:
    """A bypassed node whose matching-type input is not connected has
    nothing to forward. The frontend (ExecutableNodeDTO.resolveOutput)
    warns and drops the link: widget inputs keep their own value, optional
    connection inputs are omitted, and only a required connection input
    with no default is unrunnable.

    Found live: imgpack Basic_V38 feeds ImpactSwitch.select from a
    bypassed PrimitiveInt (which has no inputs at all), so the old
    'no matching incoming connection' error false-failed the workflow.
    """

    SWITCH_SCHEMA = {
        "ImpactSwitch": {
            "input": {
                "required": {"select": ["INT", {"default": 1}]},
                "optional": {
                    "input1": ["*"],
                    "input2": ["VAE"],
                    "input3": ["VAE"],
                },
            }
        }
    }

    def _switch_graph(self, connect_select=False, connect_vae=False):
        """ImpactSwitch(34) fed by a bypassed PrimitiveInt(73); optionally
        a bypassed VAELoader(72) into input2 (type VAE)."""
        inputs = [
            {"name": "input1", "type": "*", "link": None},
            {"name": "select", "type": "INT", "link": 79 if connect_select else None},
            {"name": "input2", "type": "VAE", "link": 78 if connect_vae else None},
        ]
        nodes = [
            node(34, "ImpactSwitch", inputs=inputs, widgets=[1, False]),
            node(73, "PrimitiveInt", mode=4,
                 outputs=[{"name": "INT", "type": "INT"}],
                 widgets=[2, "fixed"]),
        ]
        links = []
        if connect_select:
            links.append(link(79, 73, 0, 34, 1, "INT"))
        if connect_vae:
            nodes.append(node(72, "VAELoader", mode=4,
                              outputs=[{"name": "VAE", "type": "VAE"}],
                              widgets=["x.safetensors"]))
            links.append(link(78, 72, 0, 34, 2, "VAE"))
        return {"nodes": nodes, "links": links}

    def test_bypassed_primitive_feeding_widget_keeps_widget_value(self):
        # The exact live failure: select is a widget input linked from a
        # bypassed PrimitiveInt. Conversion must succeed and keep the
        # switch's own select value instead of erroring.
        schema = {**self.SWITCH_SCHEMA,
                  "PrimitiveInt": {"input": {"required": {}}}}
        api = convert_ui_workflow(self._switch_graph(connect_select=True), schema)
        assert "select" in api["34"]["inputs"]
        assert api["34"]["inputs"]["select"] == 1  # own widget value, not a link
        assert "73" not in api

    def test_bypassed_source_with_no_inputs_drops_link(self):
        # Primitive nodes have no inputs at all — the most common real case.
        schema = {**self.SWITCH_SCHEMA,
                  "PrimitiveInt": {"input": {"required": {}}}}
        api = convert_ui_workflow(self._switch_graph(connect_select=True), schema)
        assert api["34"]["inputs"]["select"] == 1

    def test_bypassed_source_optional_connection_input_is_dropped(self):
        # input2 (VAE, optional) linked from a bypassed VAELoader: the
        # link is dropped and the input simply absent from the prompt.
        schema = {**self.SWITCH_SCHEMA,
                  "VAELoader": {"input": {"required": {"vae_name": ["x.safetensors"]}}}}
        api = convert_ui_workflow(self._switch_graph(connect_vae=True), schema)
        assert "input2" not in api["34"]["inputs"]
        assert "72" not in api

    def test_bypassed_source_required_connection_without_default_is_an_error(self):
        # If the dropped input were a required connection with no default,
        # the backend could not run it — surface an actionable error.
        schema = {
            "NeedsModel": {
                "input": {
                    "required": {"model": ["MODEL"]},
                }
            }
        }
        ui = {
            "nodes": [
                node(1, "NeedsModel",
                     inputs=[{"name": "model", "type": "MODEL", "link": 50}]),
                node(2, "ModelMangler", mode=4,
                     outputs=[{"name": "MODEL", "type": "MODEL"}]),
            ],
            "links": [link(50, 2, 0, 1, 0, "MODEL")],
        }
        with pytest.raises(WorkflowConversionError) as excinfo:
            convert_ui_workflow(ui, schema)
        assert "model" in str(excinfo.value)
        assert "bypass" in str(excinfo.value).lower()


class TestClearFailures:
    def test_unknown_custom_node_names_the_node_and_suggests_custom_nodes(
        self, uma_export
    ):
        uma_export["nodes"][0]["type"] = "WanVideoSampler"
        with pytest.raises(WorkflowConversionError) as excinfo:
            convert_ui_workflow(uma_export, OBJECT_INFO)
        assert "WanVideoSampler" in str(excinfo.value)
        assert "CUSTOM_NODES" in str(excinfo.value)

    def test_link_into_muted_node_is_an_error_not_a_wrong_image(self, uma_export):
        # Uma muted her checkpoint loader (mode 2) and forgot.
        uma_export["nodes"][0]["mode"] = 2
        with pytest.raises(WorkflowConversionError) as excinfo:
            convert_ui_workflow(uma_export, OBJECT_INFO)
        assert "muted" in str(excinfo.value).lower() or "bypassed" in str(excinfo.value).lower()

    def test_missing_widget_value_falls_back_to_schema_default(self, uma_export):
        # An older export missing the trailing denoise value.
        for n in uma_export["nodes"]:
            if n["id"] == 3:
                n["widgets_values"] = [42, "randomize", 20, 7.5, "euler", "normal"]
        api = convert_ui_workflow(uma_export, OBJECT_INFO)
        assert api["3"]["inputs"]["denoise"] == 1.0

    def test_empty_workflow_is_a_clear_error(self):
        with pytest.raises(WorkflowConversionError):
            convert_ui_workflow({"nodes": [], "links": []}, OBJECT_INFO)

    def test_disconnected_unknown_type_is_skipped(self, uma_export):
        """A frontend-only UI helper (e.g. rgthree's 'Fast Groups Bypasser'
        panel, isVirtualNode) has no backend class, so it is absent from
        /object_info — yet the frontend itself keeps it out of the API
        prompt. Disconnected, it can't affect the image: skip, don't fail."""
        uma_export["nodes"].append(
            node(99, "Fast Groups Bypasser (rgthree)",
                 outputs=[{"name": "OPT_CONNECTION", "type": "*", "links": None}])
        )
        api = convert_ui_workflow(uma_export, OBJECT_INFO)
        assert "99" not in api
        assert "3" in api  # the rest of the graph is untouched

    def test_connected_unknown_type_is_a_named_error(self, uma_export):
        """An unknown type WITH connections is a genuinely missing custom
        node pack: the error must name the node and point at CUSTOM_NODES."""
        ghost = node(99, "MysteryPackNode",
                     outputs=[{"name": "MODEL", "type": "MODEL", "links": [1]}])
        # Wire the ghost into the KSampler's model input (replaces link 1,
        # whose old producer was the checkpoint loader node 4).
        for inp in next(n for n in uma_export["nodes"] if n["id"] == 3)["inputs"]:
            if inp["name"] == "model":
                inp["link"] = 1
        uma_export["nodes"].append(ghost)
        uma_export["links"].append(link(1, 99, 0, 3, 0, "MODEL"))
        with pytest.raises(WorkflowConversionError) as excinfo:
            convert_ui_workflow(uma_export, OBJECT_INFO)
        message = str(excinfo.value)
        assert "MysteryPackNode" in message
        assert "CUSTOM_NODES" in message

    def test_multiple_connected_unknowns_reported_in_one_error(self, uma_export):
        """Two missing packs should be named in a single error — one run
        surfaces every gap instead of requiring one failing job per node."""
        a = node(97, "MissingPackA",
                 outputs=[{"name": "MODEL", "type": "MODEL", "links": [11]}])
        b = node(98, "MissingPackB",
                 outputs=[{"name": "VAE", "type": "VAE", "links": [12]}])
        for nid, lname, lid in ((3, "model", 11), (8, "vae", 12)):
            for inp in next(n for n in uma_export["nodes"] if n["id"] == nid)["inputs"]:
                if inp["name"] == lname:
                    inp["link"] = lid
        uma_export["nodes"].extend([a, b])
        uma_export["links"].extend([link(11, 97, 0, 3, 0, "MODEL"),
                                    link(12, 98, 0, 8, 1, "VAE")])
        with pytest.raises(WorkflowConversionError) as excinfo:
            convert_ui_workflow(uma_export, OBJECT_INFO)
        message = str(excinfo.value)
        assert "MissingPackA" in message and "MissingPackB" in message


class TestRealCivitaiWorkflow:
    """A genuine community workflow from Civitai (FLUX img2img megapack,
    107k downloads) — vendored unmodified. Real community workflows are
    never pure-core: this one carries three custom node types, which is
    the honest common case the error path must handle well.
    """

    FIXTURE = (
        Path(__file__).resolve().parents[2]
        / "test_resources/workflows/civitai_flux_img2img_ui_format.json"
    )
    CUSTOM_NODES_IN_FIXTURE = {
        "Power Lora Loader (rgthree)",
        "SDXL Resolutions (JPS)",
        "HintImageEnchance",
    }

    # Core nodes the fixture uses that the live /object_info always carries;
    # stubbed here so the ONLY unknown nodes are the genuinely custom ones.
    CORE_FLUX_NODES = [
        "UNETLoader", "DualCLIPLoader", "VAELoader", "LoadImage", "VAEEncode",
        "ImageScaleBy", "KSamplerSelect", "BasicScheduler", "BasicGuider",
        "RandomNoise", "SamplerCustomAdvanced", "CLIPTextEncodeFlux",
    ]

    @pytest.fixture
    def schema(self):
        stubs = {n: {"input": {"required": {}}} for n in self.CORE_FLUX_NODES}
        return {**OBJECT_INFO, **stubs}

    @pytest.fixture
    def civitai_workflow(self):
        import json

        return json.loads(self.FIXTURE.read_text())

    def test_real_export_is_detected_as_ui_format(self, civitai_workflow):
        assert is_ui_format(civitai_workflow)

    def test_conversion_without_custom_nodes_names_the_missing_node(
        self, civitai_workflow, schema
    ):
        """With a core-only schema, the error must name a real custom node
        from the workflow and point at CUSTOM_NODES — that's the message a
        user needs to fix their endpoint."""
        with pytest.raises(WorkflowConversionError) as excinfo:
            convert_ui_workflow(civitai_workflow, schema)
        message = str(excinfo.value)
        assert any(name in message for name in self.CUSTOM_NODES_IN_FIXTURE)
        assert "CUSTOM_NODES" in message

    def test_fixture_reflects_community_reality(self, civitai_workflow):
        """Guard: the fixture keeps its custom nodes (nobody 'cleaned' it).
        If this fails, the fixture stopped representing real community
        workflows and the error-path test above lost its meaning."""
        types = {n.get("type") for n in civitai_workflow["nodes"]}
        assert self.CUSTOM_NODES_IN_FIXTURE <= types


class TestRealImgpackWorkflow:
    """Image Workflows 'Basic_V38' from Civitai (85k downloads) — vendored
    unmodified. The first community workflow in the gauntlet to fail on
    the converter: a bypassed PrimitiveInt feeds ImpactSwitch's 'select'
    (a widget input), which a bypassed node cannot forward. ComfyUI's
    frontend drops such links (widget keeps its own value); the converter
    must do the same instead of false-failing the whole workflow.
    """

    FIXTURE = (
        Path(__file__).resolve().parents[2]
        / "test_resources/workflows/civitai_imgpack_basic_v38_ui_format.json"
    )

    @pytest.fixture
    def imgpack_workflow(self):
        import json

        return json.loads(self.FIXTURE.read_text())

    @pytest.fixture
    def imgpack_schema(self, imgpack_workflow):
        """Hermetic schema for every node type in the fixture, derived from
        the export itself.

        For each node we emit: one schema input per *linked* input (in UI
        slot order, typed as the slot's type) plus one widget input per
        ``widgets_values`` entry (with the exported value as default). The
        converter maps ``widgets_values`` positionally against widget-type
        schema inputs, so the counts line up by construction and every
        default is present — the test isolates the bypass/drop behavior
        instead of asserting semantic value placement.
        """
        schema = {}
        for n in imgpack_workflow["nodes"]:
            ntype = n.get("type")
            if ntype in schema:
                continue
            inputs = n.get("inputs") or []
            widgets = n.get("widgets_values") or []
            required, optional = {}, {}
            # Connection inputs first (UI slot order). In real schemas,
            # widget-type linked inputs (e.g. ImpactSwitch.select, an INT
            # that can be wired or not) sit in "required" with a default —
            # which is why the frontend can drop the link and keep the
            # widget value. Pure connection inputs sit in "optional".
            for inp in inputs:
                if inp.get("link") is None:
                    continue
                name, t = inp.get("name"), inp.get("type")
                if t in ("INT", "FLOAT", "STRING", "BOOLEAN"):
                    default = {"INT": 1, "FLOAT": 0.0,
                               "STRING": "", "BOOLEAN": False}.get(t)
                    required[name] = [t, {"default": default}]
                else:
                    optional[name] = [t] if t is not None else ["*"]
            # Then widget inputs, positional with the exported values as
            # defaults (so a missing value can never abort conversion).
            for w_i, default in enumerate(widgets):
                if isinstance(default, bool):
                    required[f"w{w_i}"] = ["BOOLEAN", {"default": default}]
                elif isinstance(default, (int, float)):
                    required[f"w{w_i}"] = ["INT", {"default": default}]
                else:
                    required[f"w{w_i}"] = ["STRING", {"default": str(default)}]
            schema[ntype] = {"input": {"required": required, "optional": optional}}
        return schema

    def test_fixture_is_detected_as_ui_format(self, imgpack_workflow):
        assert is_ui_format(imgpack_workflow)

    def test_bypassed_primitive_select_converts(self, imgpack_workflow, imgpack_schema):
        """The exact live failure, whole workflow: conversion must succeed
        and the unforwardable links must drop (ImpactSwitch.select falls
        back to its own widget value, never a link reference)."""
        api = convert_ui_workflow(imgpack_workflow, imgpack_schema)
        for nid, entry in api.items():
            if entry["class_type"] == "ImpactSwitch":
                assert "select" in entry["inputs"]
                assert not isinstance(entry["inputs"]["select"], list)
        # No bypassed node appears in the executable graph.
        bypassed_ids = {str(n["id"]) for n in imgpack_workflow["nodes"] if n.get("mode") == 4}
        assert not (bypassed_ids & set(api.keys()))

    def test_fixture_keeps_its_bypassed_primitives(self, imgpack_workflow):
        """Guard: the regression only has meaning if the fixture still
        carries the bypassed PrimitiveInt -> ImpactSwitch.select links
        that failed live. If a future export 'cleans' them, this test
        fails so the fixture can be updated deliberately."""
        nodes_by_id = {str(n["id"]): n for n in imgpack_workflow["nodes"]}
        hits = 0
        for n in imgpack_workflow["nodes"]:
            if n.get("type") != "PrimitiveInt" or n.get("mode") != 4:
                continue
            for out in n.get("outputs") or []:
                for lid in out.get("links") or []:
                    link = next(l for l in imgpack_workflow["links"] if l[0] == lid)
                    consumer = nodes_by_id.get(str(link[3]))
                    if consumer and consumer.get("type") == "ImpactSwitch":
                        hits += 1
        assert hits >= 2  # nodes 34 and 26 in the original export

    def test_frontend_ghost_panel_is_skipped(self, imgpack_workflow, imgpack_schema):
        """The exact live failure, whole workflow: node 2 'Fast Groups
        Bypasser (rgthree)' is a frontend-only panel (isVirtualNode) — the
        backend never registers it, so /object_info has no entry for it.
        The frontend itself keeps such nodes out of the API prompt; the
        converter must skip it instead of failing the whole run."""
        assert "2" in {str(n["id"]) for n in imgpack_workflow["nodes"]}
        ghost_type = "Fast Groups Bypasser (rgthree)"
        assert any(n.get("type") == ghost_type for n in imgpack_workflow["nodes"])
        ghost_schema = {t: s for t, s in imgpack_schema.items() if t != ghost_type}
        api = convert_ui_workflow(imgpack_workflow, ghost_schema)
        assert "2" not in api
        # The real rgthree pack node that IS connected still converts...
        assert any(e["class_type"] == "Power Lora Loader (rgthree)"
                   for e in api.values())
