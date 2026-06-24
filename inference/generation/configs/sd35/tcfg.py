"""SD3.5-medium, Tangential-damping CFG."""

from _patch import patch_diffusers_no_bnb, SD35_MEDIUM

patch_diffusers_no_bnb()

import torch
from diffusers.pipelines.stable_diffusion_3 import pipeline_stable_diffusion_3_tcfg as sd3tcfg


def pipeline(device):
    return sd3tcfg.StableDiffusion3TCFGPipeline.from_pretrained(
        SD35_MEDIUM, torch_dtype=torch.float16
    ).to(device)


generation_params = {
    "num_inference_steps": 25,
    "guidance_scale": 6.0,
    "use_tcfg": True,
    "tcfg_rank": 2,
}
