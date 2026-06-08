"""SD3.5-medium, Perturbed Attention Guidance (factory).

Stubs soundfile before importing diffusers.pipelines.pag — that subpackage
pulls in audio pipelines that require libsndfile, missing on this server."""

import sys
import types
from importlib.machinery import ModuleSpec

if "soundfile" not in sys.modules:
    _sf = types.ModuleType("soundfile")
    _sf.__spec__ = ModuleSpec("soundfile", loader=None)
    _sf.__version__ = "0.0.0"
    _sf.read = lambda *a, **kw: (None, None)
    _sf.write = lambda *a, **kw: None
    _sf.info = lambda *a, **kw: None
    _sf.SoundFile = type("SoundFile", (), {})
    sys.modules["soundfile"] = _sf

from ._patch import patch_diffusers_no_bnb, SD35_MEDIUM

patch_diffusers_no_bnb()

import torch
from diffusers.pipelines.pag import pipeline_pag_sd_3 as sd3pag


def pipeline(device, pag_applied_layers=None):
    # `pag_applied_layers` is normally an HP from yaml but the diffusers
    # PAG pipeline takes it at construction, not at call-time. So accept it
    # here and forward; caller may also pass via yaml `generation_params`.
    if pag_applied_layers is None:
        pag_applied_layers = [r"transformer_blocks\.13\.attn$"]
    return sd3pag.StableDiffusion3PAGPipeline.from_pretrained(
        SD35_MEDIUM,
        torch_dtype=torch.float16,
        pag_applied_layers=pag_applied_layers,
    ).to(device)
