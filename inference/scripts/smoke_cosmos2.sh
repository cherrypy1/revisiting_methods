#!/usr/bin/env bash
# Smoke-test all cosmos2 methods on 2 geneval prompts each.
# Validates that every config + pipeline path actually runs.
set -euo pipefail
cd ~/geneval
module purge
module load gnu14/14.1
source ~/.venv/bin/activate
TAG=smoke_cosmos2_$(date +%d%m%Y_%H%M)
LOG_DIR=~/geneval/outputs/logs
mkdir -p "$LOG_DIR"
METHODS=(no_cfg cfg cfgpp cfg0s apg tcfg pag sag seg_sigma10 oseg)
for m in "${METHODS[@]}"; do
  echo "### smoke $m ###"
  python scripts/bench.py "$m" geneval --model cosmos2 --limit 2 --run-tag "$TAG" --skip-eval 2>&1 \
    | tee "$LOG_DIR/cosmos2_smoke_${m}.log" || echo "FAILED $m"
done
echo "DONE"
