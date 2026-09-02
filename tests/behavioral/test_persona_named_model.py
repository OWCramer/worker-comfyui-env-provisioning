"""Persona: Dana — names a model and expects the worker to work the rest out.

Dana pastes `Comfy-Org/flux1-dev` into the console. She does not know which of
that repository's files holds the weights, that FLUX samples at cfg 1, or that a
LoRA needs a checkpoint under it. None of that should be her problem, and it
should not be the console's either: the worker downloads from Hugging Face and
runs the workflows, so it is the thing that can answer.

From one repository the worker must find the weights, pick the template, and
pull whatever else that template loads.
"""

import json

import pytest

from provisioning import detect
from provisioning.errors import ResolutionError
from tests.conftest import FakeResponse


def repo(**overrides):
    base = {"tags": ["diffusers"], "siblings": [{"rfilename": "model.safetensors"}]}
    base.update(overrides)
    return base


@pytest.fixture
def hub(worker):
    """Routes repository lookups; `add` registers one by id."""

    def add(repo_id, body):
        worker.session.register(f"https://huggingface.co/api/models/{repo_id}", FakeResponse(json_data=body))

    worker.add_repo = add
    return worker


def detected(worker, source):
    return detect.detect(source, session=worker.session)


class TestCheckpoints:
    def test_reads_the_architecture_from_the_repository(self, hub):
        hub.add_repo("Comfy-Org/flux1-dev", repo(tags=["diffusers:FluxPipeline"]))

        result = detected(hub, "Comfy-Org/flux1-dev")

        assert result.template == "flux"

    def test_refuses_an_architecture_it_has_no_template_for(self, hub):
        # It says what it is, and loading it on the wrong template would render
        # nonsense rather than fail.
        hub.add_repo("someone/novel", repo(tags=["diffusers:ZImagePipeline"]))

        with pytest.raises(ResolutionError, match="ZImagePipeline"):
            detected(hub, "someone/novel")

    def test_falls_back_only_when_no_architecture_is_declared(self, hub):
        # The common shape for a community single-file merge, which
        # CheckpointLoaderSimple does load.
        hub.add_repo("someone/merge", repo(tags=["safetensors"]))

        result = detected(hub, "someone/merge")

        assert result.template == "checkpoint"
        assert any("declares no architecture" in w for w in result.warnings)

    def test_puts_the_weights_where_the_template_loads_them(self, hub):
        hub.add_repo("someone/sdxl", repo(tags=["diffusers:StableDiffusionXLPipeline"]))

        result = detected(hub, "someone/sdxl")

        assert [(m.directory, m.filename) for m in result.models] == [
            ("checkpoints", "model.safetensors")
        ]

    def test_ignores_companion_weights_beside_the_checkpoint(self, hub):
        hub.add_repo(
            "someone/flux",
            repo(
                tags=["diffusers:FluxPipeline"],
                siblings=[{"rfilename": "ae.safetensors"}, {"rfilename": "flux1-dev.safetensors"}],
            ),
        )

        result = detected(hub, "someone/flux")

        assert "flux1-dev.safetensors" in result.models[0].url

    def test_prefers_the_lower_precision_build(self, hub):
        hub.add_repo(
            "someone/flux",
            repo(
                tags=["diffusers:FluxPipeline"],
                siblings=[
                    {"rfilename": "flux1-dev-fp8.safetensors"},
                    {"rfilename": "flux1-dev.safetensors"},
                ],
            ),
        )

        result = detected(hub, "someone/flux")

        assert "fp8" in result.models[0].url

    def test_says_which_builds_it_cannot_choose_between(self, hub):
        hub.add_repo(
            "Lykon/DreamShaper",
            repo(
                siblings=[
                    {"rfilename": "DreamShaper_8_pruned.safetensors"},
                    {"rfilename": "DreamShaper_7_pruned.safetensors"},
                ]
            ),
        )

        with pytest.raises(ResolutionError, match="several builds"):
            detected(hub, "Lykon/DreamShaper")

    def test_takes_the_file_a_link_names(self, hub):
        hub.add_repo("Lykon/DreamShaper", repo(siblings=[{"rfilename": "a.safetensors"}]))

        result = detected(
            hub,
            "https://huggingface.co/Lykon/DreamShaper/resolve/main/DreamShaper_7_pruned.safetensors",
        )

        assert "DreamShaper_7_pruned.safetensors" in result.models[0].url

    def test_refuses_a_repository_with_only_sharded_weights(self, hub):
        hub.add_repo(
            "Qwen/Qwen-Image",
            repo(siblings=[{"rfilename": "model-00001-of-00002.safetensors"}]),
        )

        with pytest.raises(ResolutionError, match="no single-file"):
            detected(hub, "Qwen/Qwen-Image")


class TestLoras:
    def test_stacks_a_lora_on_the_checkpoint_its_base_declares(self, hub):
        hub.add_repo(
            "someone/a-style",
            repo(tags=["lora", "base_model:adapter:someone/private-merge"],
                 cardData={"base_model": "someone/private-merge"},
                 siblings=[{"rfilename": "style.safetensors"}]),
        )
        hub.add_repo("someone/private-merge", repo(tags=["diffusers:StableDiffusionXLPipeline"]))

        result = detected(hub, "someone/a-style")

        assert result.template == "checkpoint-lora"
        assert [(m.directory, m.filename) for m in result.models] == [
            ("checkpoints", "model.safetensors"),
            ("loras", "lora.safetensors"),
        ]

    def test_uses_the_flux_lora_template_for_a_flux_base(self, hub):
        hub.add_repo(
            "someone/a-style",
            repo(tags=["lora"], cardData={"base_model": "someone/anon-flux"},
                 siblings=[{"rfilename": "style.safetensors"}]),
        )
        hub.add_repo("someone/anon-flux", repo(tags=["diffusers:FluxPipeline"]))

        assert detected(hub, "someone/a-style").template == "flux-lora"

    def test_says_what_it_was_trained_on_when_that_is_unsupported(self, hub):
        hub.add_repo(
            "F16/krea2-turbo-sda",
            repo(tags=["lora"], cardData={"base_model": "krea/Krea-2-Turbo"},
                 siblings=[{"rfilename": "sda.safetensors"}]),
        )
        hub.add_repo("krea/Krea-2-Turbo", repo(tags=["diffusers:Krea2Pipeline"]))

        with pytest.raises(ResolutionError, match="krea/Krea-2-Turbo"):
            detected(hub, "F16/krea2-turbo-sda")

    def test_records_which_checkpoint_it_chose(self, hub):
        hub.add_repo(
            "someone/a-style",
            repo(tags=["lora"], cardData={"base_model": "someone/anon-sdxl"},
                 siblings=[{"rfilename": "style.safetensors"}]),
        )
        hub.add_repo("someone/anon-sdxl", repo(tags=["diffusers:StableDiffusionXLPipeline"]))

        result = detected(hub, "someone/a-style")

        assert any("stacking it on" in w for w in result.warnings)


class TestSplitImageFamilies:
    """Z-Image, Qwen-Image and Krea 2 split the way Wan does: a diffusion model,
    a text encoder and a VAE. Only the first is named; the rest are fixed."""

    def repackage(self, filename):
        return repo(tags=["diffusion-single-file", "comfyui"],
                    siblings=[{"rfilename": filename}])

    def test_z_image_brings_its_encoder_and_vae(self, hub):
        hub.add_repo(
            "Comfy-Org/z_image_turbo",
            self.repackage("split_files/diffusion_models/z_image_turbo_bf16.safetensors"),
        )

        result = detected(hub, "Comfy-Org/z_image_turbo")

        assert result.template == "z-image"
        assert [m.directory for m in result.models] == [
            "diffusion_models",
            "text_encoders",
            "vae",
        ]
        assert "qwen_3_4b" in result.models[1].url

    def test_qwen_image_uses_its_own_encoder(self, hub):
        hub.add_repo(
            "Comfy-Org/Qwen-Image_ComfyUI",
            self.repackage("split_files/diffusion_models/qwen_image_fp8_e4m3fn.safetensors"),
        )

        result = detected(hub, "Comfy-Org/Qwen-Image_ComfyUI")

        assert result.template == "qwen-image"
        assert "qwen_2.5_vl_7b" in result.models[1].url

    def test_krea2_is_found_in_a_top_level_directory(self, hub):
        """This repackage keeps weights in diffusion_models/, not split_files/."""
        hub.add_repo(
            "Comfy-Org/Krea-2",
            self.repackage("diffusion_models/krea2_turbo_fp8_scaled.safetensors"),
        )

        result = detected(hub, "Comfy-Org/Krea-2")

        assert result.template == "krea2"
        assert "qwen3vl_4b" in result.models[1].url

    def test_krea2_wins_over_qwen_despite_its_qwen_encoder(self, hub):
        """Both use a Qwen encoder; only the diffusion model's name separates them."""
        hub.add_repo(
            "Comfy-Org/Krea-2",
            self.repackage("diffusion_models/krea2_turbo_fp8_scaled.safetensors"),
        )

        assert detected(hub, "Comfy-Org/Krea-2").template == "krea2"

    def test_a_plain_checkpoint_is_untouched_by_this(self, hub):
        hub.add_repo(
            "someone/sdxl",
            repo(tags=["diffusers:StableDiffusionXLPipeline"],
                 siblings=[{"rfilename": "sd_xl.safetensors"}]),
        )

        result = detected(hub, "someone/sdxl")

        assert result.template == "checkpoint"
        assert len(result.models) == 1


class TestVideo:
    def test_pulls_the_encoder_and_vae_a_wan_model_needs(self, hub):
        hub.add_repo(
            "Comfy-Org/Wan_2.2_Repackaged",
            repo(
                pipeline_tag="text-to-video",
                tags=["wan2.2"],
                siblings=[
                    {"rfilename": "split_files/diffusion_models/wan2.2_ti2v_5B_fp16.safetensors"}
                ],
            ),
        )

        result = detected(hub, "Comfy-Org/Wan_2.2_Repackaged")

        assert result.template == "wan-ti2v"
        assert [m.directory for m in result.models] == [
            "diffusion_models",
            "text_encoders",
            "vae",
        ]

    def test_gives_wan_2_1_its_own_template_and_vae(self, hub):
        hub.add_repo(
            "Wan-AI/Wan2.1-T2V-1.3B",
            repo(
                pipeline_tag="text-to-video",
                tags=["wan2.1"],
                siblings=[{"rfilename": "wan2.1_t2v_1.3B_fp16.safetensors"}],
            ),
        )

        result = detected(hub, "Wan-AI/Wan2.1-T2V-1.3B")

        assert result.template == "wan-t2v"
        assert "wan_2.1_vae" in result.models[2].url

    def test_refuses_a_two_stage_wan_model(self, hub):
        hub.add_repo(
            "Comfy-Org/Wan_2.2_Repackaged",
            repo(
                pipeline_tag="text-to-video",
                tags=["wan2.2"],
                siblings=[
                    {
                        "rfilename": "split_files/diffusion_models/"
                        "wan2.2_t2v_high_noise_14B_fp8_scaled.safetensors"
                    }
                ],
            ),
        )

        with pytest.raises(ResolutionError, match="two-stage"):
            detected(hub, "Comfy-Org/Wan_2.2_Repackaged")

    def test_refuses_a_video_family_it_has_no_template_for(self, hub):
        hub.add_repo(
            "Lightricks/LTX-Video",
            repo(pipeline_tag="text-to-video", tags=["ltx"],
                 siblings=[{"rfilename": "ltx.safetensors"}]),
        )

        with pytest.raises(ResolutionError, match="no template"):
            detected(hub, "Lightricks/LTX-Video")


class TestBadInput:
    def test_rejects_a_host_it_cannot_detect_from(self, hub):
        with pytest.raises(ResolutionError, match="Hugging Face"):
            detected(hub, "https://civitai.com/models/4384/dreamshaper")

    def test_rejects_something_that_is_not_a_repository(self, hub):
        with pytest.raises(ResolutionError, match="not a Hugging Face repository"):
            detected(hub, "dreamshaper")

    def test_reports_a_model_that_is_not_there(self, hub):
        with pytest.raises(ResolutionError, match="no model"):
            detected(hub, "nobody/nothing")


class TestEndToEnd:
    def test_provisioning_downloads_everything_detection_found(self, hub):
        hub.add_repo(
            "someone/sdxl",
            repo(tags=["diffusers:StableDiffusionXLPipeline"],
                 siblings=[{"rfilename": "sd_xl.safetensors"}]),
        )
        hub.hub.publish("someone/sdxl", "sd_xl.safetensors", b"weights")
        hub.environ["COMFY_MODEL"] = "someone/sdxl"

        manifest = hub.provision(template_file=str(hub.comfy_home / "template"))

        assert hub.model_path("checkpoints", "model.safetensors").read_bytes() == b"weights"
        assert (hub.comfy_home / "template").read_text() == "checkpoint"
        assert json.loads(hub.manifest_path.read_text())["models"][0]["status"] == "cached"
