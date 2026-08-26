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
