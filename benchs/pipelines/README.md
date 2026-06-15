# Pipelines

Custom benchmark pipelines live here by model:

```text
pipelines/sd35/
pipelines/flux2_klein_base/
pipelines/cosmos2/
```

Do not patch the installed `diffusers` package for normal benchmark runs.
Configs in `configs/<model>/<method>.yaml` point to these local modules.

Flux2 Klein is loaded like the notebook workflow:

1. load `diffusers.Flux2KleinPipeline` as the base pipeline;
2. for `cfg`/`no_cfg`, use the base pipeline directly;
3. for custom methods, load the local class from `pipelines/flux2_klein_base`
   and call `PipelineClass.from_pipe(base)`.

SD3.5 and Cosmos2 currently keep their local factory modules:

```python
def pipeline(device):
    ...
```

The shared loader is `scripts/common.py`.
