#!/usr/bin/env bash
# Remaining methods for strat27042026 + apg dpg eval retry.
set -eu
cd ~/geneval
module purge
module load gnu14/14.1
source ~/.venv/bin/activate
TAG=strat27042026
LOG_DIR=~/geneval/outputs/logs
mkdir -p $LOG_DIR

run() {
  local cmd_label="$1"; shift
  local log_path="$1"; shift
  ts=$(date '+%F %T')
  echo "[$ts] ##### $cmd_label #####"
  # No pipefail; tee swallow exit. Explicit echo of result.
  if "$@" 2>&1 | tee "$log_path"; then
    echo "[$(date '+%F %T')] OK $cmd_label"
  else
    echo "[$(date '+%F %T')] FAIL $cmd_label (continuing)"
  fi
}

# Retry apg dpg eval first (network-only failure last time).
run "apg dpg eval retry" "$LOG_DIR/apg_dpg_eval_retry_${TAG}.log" \
  python scripts/eval_dpg.py --dpg-root /home/aaturevich/ELLA \
    --image-dir ~/geneval/outputs/dpg/apg_strat27042026/images \
    --pic-num 1 --resolution 1024 \
    --out-dir ~/geneval/outputs/dpg/apg_strat27042026/eval

METHODS=(no_cfg pag sag seg_sigma10_cfg3)
for m in "${METHODS[@]}"; do
  run "$m geneval" "$LOG_DIR/${m}_geneval_${TAG}.log" \
    python scripts/bench.py "$m" geneval --limit 90 --run-tag "$TAG"
  run "$m oneig" "$LOG_DIR/${m}_oneig_${TAG}.log" \
    python scripts/bench.py "$m" oneig --limit 15 --grid 1x1 --run-tag "$TAG"
  run "$m dpg" "$LOG_DIR/${m}_dpg_${TAG}.log" \
    python scripts/bench.py "$m" dpg --limit 90 --pic-num 1 --run-tag "$TAG"
done
echo "ALL DONE"
