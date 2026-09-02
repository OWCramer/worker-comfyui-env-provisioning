"""Weights a detected model needs beyond the one the user named.

A LoRA is nothing without a checkpoint under it, and a Wan diffusion model is
nothing without its text encoder and VAE. The user names one repository; these
are the files that have to come with it, kept here so the knowledge sits beside
the templates that load them.
"""

# Architecture, as Hugging Face's `diffusers:<Pipeline>` tag reports it, mapped
# to the checkpoint a LoRA for it stacks onto.
BASE_CHECKPOINTS = {
    "FluxPipeline": {
        "template": "flux",
        "url": "https://huggingface.co/Comfy-Org/flux1-dev/resolve/main/flux1-dev-fp8.safetensors",
    },
    "StableDiffusionXLPipeline": {
        "template": "checkpoint",
        "url": "https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/resolve/main/sd_xl_base_1.0.safetensors",
    },
    "StableDiffusionPipeline": {
        "template": "checkpoint",
        "url": "https://huggingface.co/Comfy-Org/stable-diffusion-v1-5-archive/resolve/main/v1-5-pruned-emaonly-fp16.safetensors",
    },
}

# Pipelines that need no companion, only the right template.
CHECKPOINT_TEMPLATES = {
    "FluxPipeline": "flux",
    "StableDiffusionXLPipeline": "checkpoint",
    "StableDiffusionXLImg2ImgPipeline": "checkpoint",
    "StableDiffusion3Pipeline": "checkpoint",
    "StableDiffusionPipeline": "checkpoint",
    "StableDiffusionImg2ImgPipeline": "checkpoint",
}

WAN_TEXT_ENCODER = "https://huggingface.co/Comfy-Org/Wan_2.1_ComfyUI_repackaged/resolve/main/split_files/text_encoders/umt5_xxl_fp8_e4m3fn_scaled.safetensors"
WAN_22_VAE = "https://huggingface.co/Comfy-Org/Wan_2.2_ComfyUI_Repackaged/resolve/main/split_files/vae/wan2.2_vae.safetensors"
WAN_21_VAE = "https://huggingface.co/Comfy-Org/Wan_2.1_ComfyUI_repackaged/resolve/main/split_files/vae/wan_2.1_vae.safetensors"

WAN_COMPANIONS = {
    "wan-ti2v": {"text_encoders": WAN_TEXT_ENCODER, "vae": WAN_22_VAE},
    "wan-t2v": {"text_encoders": WAN_TEXT_ENCODER, "vae": WAN_21_VAE},
}
