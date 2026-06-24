"""SD3.5-medium, SEG (sigma=inf, identity attention) + CFG (scale=3)."""

from _patch import patch_diffusers_no_bnb, SD35_MEDIUM

patch_diffusers_no_bnb()

import torch
from diffusers.pipelines.stable_diffusion_3 import pipeline_stable_diffusion_3_seg as sd3seg


def pipeline(device):
    return sd3seg.StableDiffusion3SegPipeline.from_pretrained(
        SD35_MEDIUM, torch_dtype=torch.float16
    ).to(device)


generation_params = {
    "num_inference_steps": 25,
    "guidance_scale": 3.0,
    "seg_scale": 3.0,
    "seg_blur_sigma": 1.0e10,
    "seg_applied_layers": ["d8"],
}
