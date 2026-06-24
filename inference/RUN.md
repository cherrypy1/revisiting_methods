# Running cfgbench campaigns

This repo uses the `cfgbench` runner as the primary entrypoint. Run commands from
the `inference/` repo root on the cluster.

## Environment

```bash
module load gnu14/14.1
source ~/_.venv/.venv/bin/activate
cd ~/inference
```

The Slurm worker preamble derives the active venv and repo path at runtime, so the
repo does not have to live at a hardcoded `~/geneval` path.

## Flux smoke

Use this before launching larger FLUX.2-klein runs:

```bash
nohup python -m cfgbench.cli run configs/campaigns/flux_smoke.yaml \
  > outputs/flux_smoke.run.log 2>&1 &
```

Monitor:

```bash
python -m cfgbench.cli status configs/campaigns/flux_smoke.yaml
python -m cfgbench.cli events configs/campaigns/flux_smoke.yaml -n 30
python -m cfgbench.cli health configs/campaigns/flux_smoke.yaml
```

## Main campaign

```bash
nohup python -m cfgbench.cli run configs/campaigns/main.yaml \
  > outputs/main.run.log 2>&1 &
```

## FLUX half-subset overnight campaigns

Run GenEval generation campaigns:

```bash
cd ~/inference
module load gnu14/14.1
source ~/_.venv/.venv/bin/activate
mkdir -p outputs

for c in no_cfg_geneval cfg_geneval pag_geneval seg_geneval cfgpp_geneval oseg_geneval tcfg_geneval cfg0s_geneval apg_geneval sag_geneval; do
  nohup ~/_.venv/.venv/bin/python -m cfgbench.cli run configs/campaigns/${c}.yaml --stage generate \
    > outputs/${c}.run.log 2>&1 &
done
```

Patch OneIG scorer once, then run OneIG generation campaigns:

```bash
cd ~/inference
module load gnu14/14.1
source ~/_.venv/.venv/bin/activate
mkdir -p outputs

# Keep OneIG's HuggingFace downloads out of the home quota.
export ONEIG_HF_HOME=${ONEIG_HF_HOME:-/tmp/$USER/hf_oneig}
export CFG_HF_HOME=$ONEIG_HF_HOME
export HF_HOME=$ONEIG_HF_HOME
export HF_HUB_DISABLE_XET=1

~/_.venv/.venv/bin/python scripts/patch_oneig_eval.py

for c in no_cfg_oneig cfg_oneig pag_oneig seg_oneig cfgpp_oneig oseg_oneig tcfg_oneig cfg0s_oneig apg_oneig; do
  nohup ~/_.venv/.venv/bin/python -m cfgbench.cli run configs/campaigns/${c}.yaml --stage generate \
    > outputs/${c}.run.log 2>&1 &
done
```

Run DPG campaigns. This requires `~/.config/cfgbench/dpg.env` with local
`MPLUG_VQA_MODEL`; see `USAGE.md` for the one-time mPLUG download and ELLA patch.

```bash
cd ~/inference
module load gnu14/14.1
source ~/_.venv/.venv/bin/activate
source ~/.config/cfgbench/dpg.env
mkdir -p outputs

i=0
for c in no_cfg_dpg pag_dpg seg_dpg cfgpp_dpg oseg_dpg tcfg_dpg cfg0s_dpg apg_dpg; do
  port=$((29500 + i))
  nohup env MPLUG_VQA_MODEL="$MPLUG_VQA_MODEL" DPG_PORT="$port" \
    ~/_.venv/.venv/bin/python -m cfgbench.cli run configs/campaigns/${c}.yaml \
    > outputs/${c}.run.log 2>&1 &
  i=$((i + 1))
done
```

Fast OneIG object-only recovery over already generated images:

```bash
cd ~/inference
bash scripts/run_oneig_object20_eval.sh
```

Stop existing `*_oneig` jobs and free cache manually first when needed. The
helper builds 20-prompt object-only eval shards from existing outputs and submits
one 2-hour Slurm worker per shard. Override with `ONEIG_OBJECT_PROMPTS=30` if
needed.

Run OneIG object-only over all already generated object prompts:

```bash
cd ~/inference
ONEIG_OBJECT_PROMPTS=all ONEIG_OBJECT_WALLTIME=06:00:00 bash scripts/run_oneig_object20_eval.sh
```

Run OneIG text-only over all already generated text prompts. This scorer uses
`Qwen/Qwen2.5-VL-7B-Instruct`, so keep the Qwen HuggingFace cache until text
validation is finished. If the home quota is tight, point `HF_HOME` to scratch
or node-local temporary storage before submitting:

```bash
cd ~/inference
HF_HOME=/tmp/$USER/hf_oneig_text ONEIG_TEXT_PROMPTS=all ONEIG_TEXT_WALLTIME=08:00:00 \
  bash scripts/run_oneig_text_eval.sh
```

Run OneIG reasoning-only over all already generated reasoning prompts. Reasoning
does not use Qwen; it uses LLM2CLIP checkpoints:
`openai/clip-vit-large-patch14-336`, `microsoft/LLM2CLIP-Openai-L-14-336`,
and `microsoft/LLM2CLIP-Llama-3-8B-Instruct-CC-Finetuned`.
On V100, the helper applies `scripts/patch_oneig_reasoning_v100.py` inside each
Slurm job to replace LLM2Vec's FlashAttention2-only guard with an eager fallback.

```bash
cd ~/inference
HF_HOME=/tmp/$USER/hf_oneig_reasoning ONEIG_REASONING_PROMPTS=all ONEIG_REASONING_WALLTIME=08:00:00 \
  bash scripts/run_oneig_reasoning_eval.sh
```

Monitor:

```bash
for c in no_cfg_dpg pag_dpg seg_dpg cfgpp_dpg oseg_dpg tcfg_dpg cfg0s_dpg apg_dpg \
         no_cfg_geneval cfg_geneval pag_geneval seg_geneval cfgpp_geneval oseg_geneval tcfg_geneval cfg0s_geneval apg_geneval sag_geneval \
         no_cfg_oneig cfg_oneig pag_oneig seg_oneig cfgpp_oneig oseg_oneig tcfg_oneig cfg0s_oneig apg_oneig; do
  ~/_.venv/.venv/bin/python -m cfgbench.cli status configs/campaigns/${c}.yaml
done
```

Build report after completion:

```bash
python -m cfgbench.cli report configs/campaigns/main.yaml
```

## Prompt inspection

```bash
python -m cfgbench.cli prompts geneval -n 5
python -m cfgbench.cli prompts oneig -n 5
python -m cfgbench.cli prompts dpg -n 5
```

## External assets

Defaults:

- `GENEVAL_ROOT=$HOME/geneval-bench`
- `ONEIG_ROOT=$HOME/OneIG-Benchmark`
- `DPG_ROOT=$HOME/ELLA`
- `FLUX2_MODEL_ID=black-forest-labs/FLUX.2-klein-base-4B`

Override any of them via env vars if your cluster layout differs.
