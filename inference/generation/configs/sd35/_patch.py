"""Shared helpers for SD3.5 pipeline configs.

Remote server lacks bitsandbytes. Patch `is_bitsandbytes_available` to
avoid import failures when diffusers tries to load quantized pipelines.
"""

import importlib


def patch_diffusers_no_bnb():
    for module_name in ("diffusers.utils.import_utils", "diffusers.utils"):
        module = importlib.import_module(module_name)
        module.is_bitsandbytes_available = lambda: False


SD35_MEDIUM = "stabilityai/stable-diffusion-3.5-medium"
