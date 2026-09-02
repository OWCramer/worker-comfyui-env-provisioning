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

_HF = "https://huggingface.co"
QWEN_IMAGE_VAE = f"{_HF}/Comfy-Org/Qwen-Image_ComfyUI/resolve/main/split_files/vae/qwen_image_vae.safetensors"

# Families that split across a diffusion model, a text encoder and a VAE, the
# way Wan does. Only the diffusion model is named by the user; the rest are
# fixed, so they live here beside the templates that load them.
#
# Matched on the diffusion model's filename rather than the repository's
# pipeline tag: the repackaged repositories that carry single-file weights
# declare no pipeline at all, while the originals that declare one ship only
# diffusers-format directories nothing here can load.
SPLIT_FAMILIES = [
    {
        "match": r"krea2",
        "template": "krea2",
        "text_encoders": f"{_HF}/Comfy-Org/Krea-2/resolve/main/text_encoders/qwen3vl_4b_fp8_scaled.safetensors",
        "vae": QWEN_IMAGE_VAE,
    },
    {
        "match": r"z[_-]?image",
        "template": "z-image",
        "text_encoders": f"{_HF}/Comfy-Org/z_image_turbo/resolve/main/split_files/text_encoders/qwen_3_4b.safetensors",
        "vae": f"{_HF}/Comfy-Org/z_image_turbo/resolve/main/split_files/vae/ae.safetensors",
    },
    {
        "match": r"qwen[_-]?image",
        "template": "qwen-image",
        "text_encoders": f"{_HF}/Comfy-Org/Qwen-Image_ComfyUI/resolve/main/split_files/text_encoders/qwen_2.5_vl_7b_fp8_scaled.safetensors",
        "vae": QWEN_IMAGE_VAE,
    },
]
