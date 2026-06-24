#!/usr/bin/env bash
# Final-eval runner: 4 methods × 3 benches, budget ≤1000 imgs total.
#
# Per method:
#   GenEval  : --limit 100, n_samples=1 (config default) = 100 imgs
#   OneIG    : --limit 15 per-cat × 3 cats × grid=1x1     = 45 imgs
#   DPG      : --limit 100, --pic-num 1                   = 100 imgs
# Total = 245 imgs/method × 4 methods = 980 ≤ 1000.
#
# bench.py triggers the bench-specific eval right after gen (skip with --skip-eval).
set -euo pipefail

cd ~/geneval
module purge
module load gnu14/14.1
source ~/.venv/bin/activate

TAG=final22042026
METHODS=(cfgpp cfg0s oseg tcfg)

ts() { date '+%F %T'; }
say() { echo "[$(ts)] $*"; }

for m in "${METHODS[@]}"; do
  say "=== $m :: geneval ==="
  python scripts/bench.py "$m" geneval --limit 100 --run-tag "$TAG"

  say "=== $m :: oneig ==="
  python scripts/bench.py "$m" oneig  --limit 15  --grid 1x1 --run-tag "$TAG"

  say "=== $m :: dpg ==="
  python scripts/bench.py "$m" dpg    --limit 100 --pic-num 1 --run-tag "$TAG"
done

say "=== ALL BENCHES DONE ==="
