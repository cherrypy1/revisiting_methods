"""SD3.5-medium, CFG++ (flow-matching variant)."""

from _patch import patch_diffusers_no_bnb, SD35_MEDIUM

patch_diffusers_no_bnb()

import torch
from diffusers.pipelines.stable_diffusion_3 import pipeline_stable_diffusion_3_cfgpp as sd3cfgpp


def pipeline(device):
    return sd3cfgpp.StableDiffusion3CFGPPPipeline.from_pretrained(
        SD35_MEDIUM, torch_dtype=torch.float16
    ).to(device)


# CFG++ for rectified flow reparametrizes guidance as x0_guided =
# x0_uncond + w * (x0_text - x0_uncond), so `w` lives in [~0.5, ~1.3] and
# behaves like the paper's mixing coefficient (NOT a standard CFG scale —
# using w=5.5 produces the burnout artifacts observed empirically).
generation_params = {
    "num_inference_steps": 25,
    "guidance_scale": 1.2,
}
