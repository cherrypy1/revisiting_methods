#!/usr/bin/env bash
# Runs all 1D sweeps sequentially on a single GPU.
# Uses 20 prompts from prompts/evaluation_metadata.jsonl.
set -euo pipefail

cd ~/geneval
module purge
module load gnu14/14.1
source ~/.venv/bin/activate

PROMPTS=prompts/evaluation_metadata.jsonl
N=20
OUT=outputs/sweep
mkdir -p "$OUT"

ts() { date '+%F %T'; }
say() { echo "[$(ts)] $*"; }

say "=== cfgpp guidance_scale ==="
python scripts/sweep.py generation/configs/sd35/cfgpp.py \
    --param guidance_scale --values 0.5 0.7 0.9 1.2 \
    --prompts "$PROMPTS" --n-prompts $N --out-dir "$OUT/cfgpp_scale"

say "=== cfg0s guidance_scale ==="
python scripts/sweep.py generation/configs/sd35/cfg0s.py \
    --param guidance_scale --values 4.5 5.5 6.5 \
    --prompts "$PROMPTS" --n-prompts $N --out-dir "$OUT/cfg0s_scale"

say "=== cfg0s zero_steps ==="
python scripts/sweep.py generation/configs/sd35/cfg0s.py \
    --param zero_steps --values 0 1 2 \
    --prompts "$PROMPTS" --n-prompts $N --out-dir "$OUT/cfg0s_zsteps"

say "=== oseg oseg_scale ==="
python scripts/sweep.py generation/configs/sd35/oseg.py \
    --param oseg_scale --values 0.5 1.0 1.5 2.0 \
    --prompts "$PROMPTS" --n-prompts $N --out-dir "$OUT/oseg_oscale"

say "=== oseg seg_scale ==="
python scripts/sweep.py generation/configs/sd35/oseg.py \
    --param seg_scale --values 2.0 3.0 4.0 \
    --prompts "$PROMPTS" --n-prompts $N --out-dir "$OUT/oseg_sscale"

say "=== tcfg tcfg_rank ==="
python scripts/sweep.py generation/configs/sd35/tcfg.py \
    --param tcfg_rank --values 1 2 \
    --prompts "$PROMPTS" --n-prompts $N --out-dir "$OUT/tcfg_rank"

say "=== tcfg guidance_scale ==="
python scripts/sweep.py generation/configs/sd35/tcfg.py \
    --param guidance_scale --values 4.0 5.5 7.0 \
    --prompts "$PROMPTS" --n-prompts $N --out-dir "$OUT/tcfg_scale"

say "=== pag pag_scale ==="
python scripts/sweep.py generation/configs/sd35/pag.py \
    --param pag_scale --values 1.0 2.0 3.0 5.0 \
    --prompts "$PROMPTS" --n-prompts $N --out-dir "$OUT/pag_scale"

say "=== ALL SWEEPS DONE ==="
