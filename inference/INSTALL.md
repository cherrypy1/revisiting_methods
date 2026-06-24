# INSTALL — cfgbench on the cHARISMa cluster

Reproduces the validated environment + benchmark assets for another account on the **same
cluster**. Most of it is automated by `scripts/install.py`; a few things need manual steps
(HuggingFace licenses, the bitsandbytes wheel, a DPG-eval patch).

Paths are all `$HOME`-relative — no hardcoded user. The repo can live anywhere (the
orchestrator derives its venv + repo path at runtime); `~/geneval` is just the convention.

## Prerequisites

- cHARISMa account with GPU access (partition `rocky`, V100 32GB). If your profile needs a
  SLURM account, set `pool.account` in the campaign yaml.
- `module load gnu14/14.1` available (newer glibc — needed for torch + bitsandbytes).
- Internet on the login node (git clone + pip + HF/openmmlab downloads).

## Quick start

```bash
git clone <this repo> ~/geneval && cd ~/geneval
module load gnu14/14.1
python -m venv ~/.venv && source ~/.venv/bin/activate
python scripts/install.py            # deps + repos + detector weights + shims + verify
```

Then the **manual** steps `install.py` prints:

1. **HuggingFace** (generation weights, gated):
   ```bash
   huggingface-cli login
   ```
   Accept licenses: `stabilityai/stable-diffusion-3.5-medium`,
   `nvidia/Cosmos-Predict2-2B-Text2Image`. Weights pull on first run into `~/.cache/huggingface`.
2. **bitsandbytes 0.43.2** — *Cosmos2 only* (T5 4-bit nf4). The manylinux wheel is rejected by
   the default glibc; download + retag to a plain `linux_x86_64` wheel, then `pip install` it.
   Skip if you only run SD3.5.
3. **DPG eval offline patch** — modelscope 1.35.3 checks revision online and times out; patch
   `~/ELLA/dpg_bench/compute_dpg_bench.py` to load mPLUG from a local cache. Skip if you don't
   run DPG.

Optional — Telegram alerts:
```bash
mkdir -p ~/.config/cfgbench
printf 'export TELEGRAM_BOT_TOKEN=...\nexport TELEGRAM_CHAT_ID=...\n' > ~/.config/cfgbench/notify.env
python -m cfgbench.cli notify-test
```

## Verify

```bash
python scripts/install.py --only verify           # libs + assets + imports
python -m cfgbench.cli prompts geneval            # 553; oneig 631; dpg 1065
```

## What lands where

| Path | What | Installed by |
|------|------|--------------|
| `~/.venv` | python deps (torch 2.5.1+cu121, diffusers@`fa468c5d5`, transformers 4.57.3, mmcv/mmdet, …) | `install.py --only deps` |
| `~/geneval-bench` | GenEval scorer (`evaluation/`, `prompts/`) + `mmdetection/` config + `models/` Mask2Former | `install.py` (repos + weights) |
| `~/OneIG-Benchmark` | OneIG repo + `OneIG-Bench.csv` (VLM scorers download on first eval) | `install.py` |
| `~/ELLA` | DPG-Bench (`dpg_bench/prompts/`, `compute_dpg_bench.py`; mPLUG downloads on first eval) | `install.py` |
| `~/.cache/huggingface` | SD3.5 / Cosmos2 generation weights | first run (after HF login) |

cfgbench finds these via env defaults (`GENEVAL_ROOT`/`ONEIG_ROOT`/`DPG_ROOT` = `~/geneval-bench` etc.);
override with env vars if you put them elsewhere.

## Which benches need what

- **GenEval**: mmdet stack + Mask2Former + `~/geneval-bench`. No bitsandbytes.
- **OneIG**: `~/OneIG-Benchmark` + its VLM scorers (auto-download). No bitsandbytes.
- **DPG**: `~/ELLA` + mPLUG (modelscope) + the offline patch + the soundfile shim (auto-written).
- **Cosmos2 generation**: bitsandbytes 0.43.2 wheel (SD3.5 generation does not need it).

## Run

See **USAGE.md**. In short, on the login node under tmux:
```bash
python -m cfgbench.cli run configs/campaigns/main.yaml      # add limit: for a fast pass
python -m cfgbench.cli status configs/campaigns/main.yaml
```

Adding a model/benchmark: **ADAPTERS.md** (one file + one registry line).
