#!/usr/bin/env bash
# Full Cosmos2 run: 10 methods x 3 benches. Same limits as SD3.5 strat.
set -eu
cd ~/geneval
module purge
module load gnu14/14.1
source ~/.venv/bin/activate
TAG=cosmos2_$(date +%d%m%Y)
LOG_DIR=~/geneval/outputs/logs
mkdir -p $LOG_DIR

run() {
  local label="$1"; shift
  local log="$1"; shift
  ts=$(date '+%F %T')
  echo "[$ts] ##### $label #####"
  if "$@" 2>&1 | tee "$log"; then
    echo "[$(date '+%F %T')] OK $label"
  else
    echo "[$(date '+%F %T')] FAIL $label (continuing)"
  fi
}

METHODS=(cfg no_cfg cfgpp cfg0s apg tcfg sag seg_sigma10 oseg pag)
for m in "${METHODS[@]}"; do
  run "$m geneval" "$LOG_DIR/cosmos2_${m}_geneval_${TAG}.log" \
    python scripts/bench.py "$m" geneval --model cosmos2 --limit 90 --run-tag "$TAG"
  run "$m oneig" "$LOG_DIR/cosmos2_${m}_oneig_${TAG}.log" \
    python scripts/bench.py "$m" oneig --model cosmos2 --limit 15 --grid 1x1 --run-tag "$TAG"
  run "$m dpg" "$LOG_DIR/cosmos2_${m}_dpg_${TAG}.log" \
    python scripts/bench.py "$m" dpg --model cosmos2 --limit 90 --pic-num 1 --run-tag "$TAG"
done
echo "ALL DONE"
