"""SD3.5-medium, Perturbed Attention Guidance (diffusers built-in)."""

import sys
import types
from importlib.machinery import ModuleSpec

# `diffusers.pipelines.pag` pulls in audio pipelines that import soundfile.
# Server lacks libsndfile.so; stub the module before the import below.
if "soundfile" not in sys.modules:
    _sf = types.ModuleType("soundfile")
    _sf.__spec__ = ModuleSpec("soundfile", loader=None)
    _sf.__version__ = "0.0.0"
    _sf.read = lambda *a, **kw: (None, None)
    _sf.write = lambda *a, **kw: None
    _sf.info = lambda *a, **kw: None
    _sf.SoundFile = type("SoundFile", (), {})
    sys.modules["soundfile"] = _sf

from _patch import patch_diffusers_no_bnb, SD35_MEDIUM

patch_diffusers_no_bnb()

import torch
from diffusers.pipelines.pag import pipeline_pag_sd_3 as sd3pag


def pipeline(device):
    # SD3.5-medium has both joint `attn` and self-only `attn2` per block.
    # Built-in PAG joint processor needs encoder_hidden_states, so match
    # only the joint attention via end-anchored regex.
    return sd3pag.StableDiffusion3PAGPipeline.from_pretrained(
        SD35_MEDIUM,
        torch_dtype=torch.float16,
        pag_applied_layers=[r"transformer_blocks\.13\.attn$"],
    ).to(device)


generation_params = {
    "num_inference_steps": 25,
    "guidance_scale": 4.5,
    "pag_scale": 3.0,
}
