"""SD3.5-medium, Adaptive Projected Guidance."""

from _patch import patch_diffusers_no_bnb, SD35_MEDIUM

patch_diffusers_no_bnb()

import torch
from diffusers.pipelines.stable_diffusion_3 import pipeline_stable_diffusion_3_apg as sd3apg


def pipeline(device):
    return sd3apg.StableDiffusion3APGPipeline.from_pretrained(
        SD35_MEDIUM, torch_dtype=torch.float16
    ).to(device)


# APG paper defaults: eta=0, norm_threshold=15, momentum=-0.5
generation_params = {
    "num_inference_steps": 25,
    "guidance_scale": 7.0,
    "apg_eta": 0.0,
    "apg_step_radius": 15.0,
    "apg_momentum": -0.5,
}
