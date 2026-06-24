"""SD3.5-medium, CFG-Zero* (Fang et al. 2025)."""

from _patch import patch_diffusers_no_bnb, SD35_MEDIUM

patch_diffusers_no_bnb()

import torch
from diffusers.pipelines.stable_diffusion_3 import pipeline_stable_diffusion_3_cfg0s as sd3cfg0s


def pipeline(device):
    return sd3cfg0s.StableDiffusion3CFG0SPipeline.from_pretrained(
        SD35_MEDIUM, torch_dtype=torch.float16
    ).to(device)


generation_params = {
    "num_inference_steps": 25,
    "guidance_scale": 6.5,
    "use_zero_init": True,
    # impl has off-by-one: `i <= zero_steps` means zero_steps=0 gives 1 zero step
    # (matches paper K=1 for SD3). Non-zero values produced artifacts in sweep.
    "zero_steps": 0,
}
