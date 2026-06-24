#!/usr/bin/env bash
# Cosmos2 geneval, 30 prompts per method, 9 methods (no pag).
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

# cheap to expensive
METHODS=(cfg no_cfg cfgpp cfg0s apg tcfg sag oseg seg_sigma10)
for m in "${METHODS[@]}"; do
  run "cosmos2 $m geneval" "$LOG_DIR/cosmos2_${m}_geneval_${TAG}.log" \
    python scripts/bench.py "$m" geneval --model cosmos2 --limit 30 --run-tag "$TAG"
done
echo "ALL DONE"
