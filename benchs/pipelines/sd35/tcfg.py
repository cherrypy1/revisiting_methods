"""SD3.5-medium, tcfg (factory) — uses consolidated methods pipeline."""

from ._patch import patch_diffusers_no_bnb, SD35_MEDIUM

patch_diffusers_no_bnb()

import torch
from diffusers.pipelines.stable_diffusion_3 import pipeline_stable_diffusion_3_methods as sd3m


def pipeline(device):
    return sd3m.StableDiffusion3MethodsPipeline.from_pretrained(
        SD35_MEDIUM, torch_dtype=torch.float16
    ).to(device)
