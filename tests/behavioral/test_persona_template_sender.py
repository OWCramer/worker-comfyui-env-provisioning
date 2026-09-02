"""Persona: Tom — deployed a model and wants to send it a prompt.

Tom picked a model in the console and got an endpoint. He has never seen a
ComfyUI graph and should not have to: the image carries the workflow, the deploy
named which one in COMFY_TEMPLATE, and his request is a prompt.

    {"input": {"prompt": "a lighthouse at sunrise"}}

He can override any parameter the template exposes, and anyone still sending a
full graph must be unaffected.
"""

import pytest

import handler
import templates


@pytest.fixture
def checkpoint_env(monkeypatch):
    monkeypatch.setenv("COMFY_TEMPLATE", "checkpoint")


class TestSendingParameters:
    def test_a_prompt_alone_is_a_valid_request(self, checkpoint_env):
        validated, error = handler.validate_input({"prompt": "a lighthouse at sunrise"})

        assert error is None
        graph = validated["workflow"]
        prompts = [
            node["inputs"]["text"]
            for node in graph.values()
            if node["class_type"] == "CLIPTextEncode"
        ]
        assert "a lighthouse at sunrise" in prompts

    def test_an_empty_request_still_renders_something(self, checkpoint_env):
        validated, error = handler.validate_input({})

        assert error is None
        assert validated["workflow"]["5"]["class_type"] == "KSampler"

    def test_overriding_a_number_keeps_it_a_number(self, checkpoint_env):
        validated, _ = handler.validate_input({"steps": 8})

        steps = validated["workflow"]["5"]["inputs"]["steps"]
        assert steps == 8
        assert isinstance(steps, int)

    def test_untouched_parameters_keep_the_template_default(self, checkpoint_env):
        validated, _ = handler.validate_input({"steps": 8})

        assert validated["workflow"]["5"]["inputs"]["cfg"] == 7

    def test_a_typo_says_what_the_template_accepts(self, checkpoint_env):
        validated, error = handler.validate_input({"stpes": 8})

        assert validated is None
        assert "stpes" in error
        assert "steps" in error


class TestDeployTimeDefaults:
    """The deploy can bake in per-model values the template cannot know."""

    def test_defaults_from_the_endpoint_apply(self, checkpoint_env, monkeypatch):
        monkeypatch.setenv("COMFY_TEMPLATE_DEFAULTS", '{"steps": 12, "prompt": "advt, a castle"}')

        validated, error = handler.validate_input({})

        assert error is None
        assert validated["workflow"]["5"]["inputs"]["steps"] == 12
        assert validated["workflow"]["2"]["inputs"]["text"] == "advt, a castle"

    def test_the_request_still_wins_over_them(self, checkpoint_env, monkeypatch):
        monkeypatch.setenv("COMFY_TEMPLATE_DEFAULTS", '{"steps": 12}')

        validated, _ = handler.validate_input({"steps": 30})

        assert validated["workflow"]["5"]["inputs"]["steps"] == 30

    def test_malformed_defaults_are_reported_not_ignored(self, checkpoint_env, monkeypatch):
        monkeypatch.setenv("COMFY_TEMPLATE_DEFAULTS", "not json")

        validated, error = handler.validate_input({"prompt": "x"})

        assert validated is None
        assert "COMFY_TEMPLATE_DEFAULTS" in error


class TestOtherTemplates:
    def test_flux_samples_at_cfg_one(self, monkeypatch):
        monkeypatch.setenv("COMFY_TEMPLATE", "flux")

        validated, _ = handler.validate_input({"prompt": "a terrarium"})

        assert validated["workflow"]["6"]["inputs"]["cfg"] == 1
        assert validated["workflow"]["3"]["class_type"] == "FluxGuidance"

    def test_a_lora_template_weights_model_and_clip_together(self, monkeypatch):
        monkeypatch.setenv("COMFY_TEMPLATE", "checkpoint-lora")

        validated, _ = handler.validate_input({"lora_strength": 0.7})

        lora = validated["workflow"]["lora"]["inputs"]
        assert lora["strength_model"] == 0.7
        assert lora["strength_clip"] == 0.7

    def test_video_ends_in_a_save_video_node(self, monkeypatch):
        monkeypatch.setenv("COMFY_TEMPLATE", "wan-ti2v")

        validated, _ = handler.validate_input({"prompt": "a subway station"})

        classes = [node["class_type"] for node in validated["workflow"].values()]
        assert "SaveVideo" in classes


class TestUnchangedBehaviour:
    def test_an_explicit_graph_wins_over_the_template(self, checkpoint_env):
        graph = {"9": {"class_type": "SaveImage", "inputs": {"images": ["1", 0]}}}

        validated, error = handler.validate_input({"workflow": graph})

        assert error is None
        assert validated["workflow"] == graph

    def test_without_a_template_a_graph_is_still_required(self):
        validated, error = handler.validate_input({"prompt": "a lighthouse"})

        assert validated is None
        assert "Missing 'workflow' parameter" in error

    def test_an_unknown_template_names_the_ones_that_exist(self, monkeypatch):
        monkeypatch.setenv("COMFY_TEMPLATE", "nope")

        validated, error = handler.validate_input({"prompt": "x"})

        assert validated is None
        assert "checkpoint" in error


class TestEveryShippedTemplate:
    """Whatever a template's shape, it must build and hang together."""

    @pytest.mark.parametrize("name", templates.available())
    def test_builds_from_its_own_defaults(self, name, monkeypatch):
        monkeypatch.setenv("COMFY_TEMPLATE", name)

        validated, error = handler.validate_input({})

        assert error is None, error
        graph = validated["workflow"]
        ids = set(graph)
        for node in graph.values():
            for value in node["inputs"].values():
                if isinstance(value, list) and len(value) == 2 and isinstance(value[0], str):
                    assert value[0] in ids, f"{name} references missing node {value[0]}"

    @pytest.mark.parametrize("name", templates.available())
    def test_leaves_no_placeholder_unfilled(self, name, monkeypatch):
        monkeypatch.setenv("COMFY_TEMPLATE", name)

        validated, _ = handler.validate_input({})

        assert "{{" not in str(validated["workflow"])
