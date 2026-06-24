#!/usr/bin/env bash
# Stratified-random round: 10 methods × {geneval,oneig,dpg}
# Limits: geneval 90 (15/cat × 6), oneig 15/cat × 3, dpg 90.
set -euo pipefail
cd ~/geneval
module purge
module load gnu14/14.1
source ~/.venv/bin/activate
TAG=strat27042026
METHODS=(cfg cfgpp cfg0s oseg tcfg apg no_cfg pag sag seg_sigma10_cfg3)
for m in "${METHODS[@]}"; do
  echo "##### $m #####"
  python scripts/bench.py "$m" geneval --limit 90 --run-tag "$TAG"
  python scripts/bench.py "$m" oneig  --limit 15 --grid 1x1 --run-tag "$TAG"
  python scripts/bench.py "$m" dpg    --limit 90 --pic-num 1 --run-tag "$TAG"
done
echo "ALL DONE"
