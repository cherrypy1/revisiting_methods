"""Shared helpers for SD3.5 pipeline configs.

Patch `is_bitsandbytes_available` to avoid import failures when diffusers tries to load quantized pipelines.
"""

import importlib


def patch_diffusers_no_bnb():
    # Force diffusers to treat bitsandbytes as unavailable (old bnb + cluster glibc
    # break diffusers' own bnb integration). Replicates the import_utils edit that
    # used to live in the editable diffusers checkout, so STOCK diffusers works too.
    iu = importlib.import_module("diffusers.utils.import_utils")
    iu._bitsandbytes_available = False
    iu._bitsandbytes_version = ""
    for module_name in ("diffusers.utils.import_utils", "diffusers.utils"):
        module = importlib.import_module(module_name)
        module.is_bitsandbytes_available = lambda: False


SD35_MEDIUM = "stabilityai/stable-diffusion-3.5-medium"
