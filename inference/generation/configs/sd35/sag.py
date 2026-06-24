"""SD3.5-medium, Self-Attention Guidance + CFG."""

from _patch import patch_diffusers_no_bnb, SD35_MEDIUM

patch_diffusers_no_bnb()

import torch
from diffusers.pipelines.stable_diffusion_3 import pipeline_stable_diffusion_3_sag as sd3sag


def pipeline(device):
    return sd3sag.StableDiffusion3SagPipeline.from_pretrained(
        SD35_MEDIUM, torch_dtype=torch.float16
    ).to(device)


generation_params = {
    "num_inference_steps": 25,
    "guidance_scale": 5.0,
    "sag_scale": 0.4,
    "sag_applied_layers": ["d0"],
}
