# Benchmarking Training-Free Guidance Methods

This repository contains pipelines, optimal configs , benchmark code and scripts for training-free guidance methods for text-to-image
diffusion models.

The project evaluates CFG-style and attention-guidance methods on modern
open-weight models, primarily Stable Diffusion 3.5 Medium and FLUX.2 Klein 4b, using
GenEval, DPG-Bench, and OneIG-Bench.

## Methods

Implemented methods include:

- no-CFG
- CFG
- CFG++
- CFG-Zero*
- APG
- TCFG
- SAG
- SEG / OSEG
- PAG

## Repository Layout

- `inference/` - benchmark runner, model pipelines, configs, Slurm scripts, and
  setup notes.
- `inference/configs/` - method configs and campaign definitions.
- `inference/pipelines/` - custom method pipelines.

-`PROGRESS.md` - actual benchmark results
## Running Benchmarks

The main runner lives under `inference/`.

```bash
cd inference
python -m cfgbench.cli run configs/campaigns/main.yaml
python -m cfgbench.cli status configs/campaigns/main.yaml
python -m cfgbench.cli report configs/campaigns/main.yaml
```

See `inference/INSTALL.md` for environment setup and `inference/RUN.md` for the
cluster commands used in the experiments.
