"""SD3.5-medium, vanilla CFG."""

from _patch import patch_diffusers_no_bnb, SD35_MEDIUM

patch_diffusers_no_bnb()

import torch
from diffusers import StableDiffusion3Pipeline


def pipeline(device):
    return StableDiffusion3Pipeline.from_pretrained(
        SD35_MEDIUM, torch_dtype=torch.float16
    ).to(device)


generation_params = {
    "num_inference_steps": 25,
    "guidance_scale": 5.5,
}
