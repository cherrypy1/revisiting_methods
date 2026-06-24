"""SD3.5-medium, OSEG (orthogonalised SEG) + CFG."""

from _patch import patch_diffusers_no_bnb, SD35_MEDIUM

patch_diffusers_no_bnb()

import torch
from diffusers.pipelines.stable_diffusion_3 import pipeline_stable_diffusion_3_oseg as sd3oseg


def pipeline(device):
    return sd3oseg.StableDiffusion3OSEGPipeline.from_pretrained(
        SD35_MEDIUM, torch_dtype=torch.float16
    ).to(device)


generation_params = {
    "num_inference_steps": 25,
    "guidance_scale": 3.0,
    "seg_scale": 2.0,
    "oseg_scale": 0.5,
    "seg_blur_sigma": 1.0e10,
    "seg_applied_layers": ["d8"],
}
