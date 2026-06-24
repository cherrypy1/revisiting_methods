#!/usr/bin/env bash
# Continue strat27042026: 9 methods remaining (cfg already done).
set -euo pipefail
cd ~/geneval
module purge
module load gnu14/14.1
source ~/.venv/bin/activate
TAG=strat27042026
METHODS=(cfgpp cfg0s oseg tcfg apg no_cfg pag sag seg_sigma10_cfg3)
LOG_DIR=~/geneval/outputs/logs
mkdir -p $LOG_DIR
for m in "${METHODS[@]}"; do
  ts=$(date '+%F %T')
  echo "[$ts] ##### $m #####"
  python scripts/bench.py "$m" geneval --limit 90 --run-tag "$TAG" 2>&1 | tee "$LOG_DIR/${m}_geneval_${TAG}.log"
  python scripts/bench.py "$m" oneig  --limit 15 --grid 1x1 --run-tag "$TAG" 2>&1 | tee "$LOG_DIR/${m}_oneig_${TAG}.log"
  python scripts/bench.py "$m" dpg    --limit 90 --pic-num 1 --run-tag "$TAG" 2>&1 | tee "$LOG_DIR/${m}_dpg_${TAG}.log"
done
echo "ALL DONE"
