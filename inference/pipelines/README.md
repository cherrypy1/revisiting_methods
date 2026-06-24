# Pipelines

Two layers live here:

1. **Custom diffusers pipeline implementations** — `pipeline_stable_diffusion_3_*.py`
   in `sd35/`, and `pipeline_cosmos2_methods.py` in `cosmos2/`. These are the
   source files that get deployed into the remote patched `diffusers` checkout
   at `~/diffusers/src/diffusers/pipelines/{stable_diffusion_3,…}/`. After
   editing, push them out with:

       python scripts/sync_pipelines.py --host hse-hpc

   Once synced, code that does
   `from diffusers.pipelines.stable_diffusion_3 import pipeline_stable_diffusion_3_X`
   picks up the new version.

2. **Thin factory modules** — `sd35/<method>.py` and `cosmos2/<method>.py`.
   Each exposes one function:

       def pipeline(device): ...

   that builds the pipeline (loading weights, applying any monkeypatches like
   `_patch.patch_diffusers_no_bnb` or the `soundfile` stub). The yaml configs
   in `configs/<model>/<method>.yaml` reference these factories via the
   `pipeline:` field as a Python module path:

       pipeline: pipelines.sd35.pag
       generation_params: {num_inference_steps: 25, guidance_scale: 4.5, ...}

   The loader in `scripts/common.py` imports the module, calls its `pipeline()`
   factory, and merges `generation_params` from the yaml into the runtime call
   kwargs.
