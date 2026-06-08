#!/usr/bin/env bash
# Cosmos2 bench remainder (run on the allocated compute node).
# Phase 1: fill geneval gaps (seg_sigma10 redo 4->30, pag 0->30).
# Phase 2: OneIG all 10 methods (limit 15, grid 1x1).
# Phase 3: DPG all 10 methods (limit 90, pic-num 1).
# Reuses existing tag so outputs land beside the 8 done geneval methods.
set -u
cd ~/geneval
module purge
module load gnu14/14.1
source ~/.venv/bin/activate

TAG=cosmos2_24052026
ALL="cfg no_cfg cfgpp cfg0s apg tcfg sag seg_sigma10 oseg pag"

echo "########## PHASE 1: geneval gaps ##########"
python scripts/run_methods.py --model cosmos2 \
  --methods seg_sigma10 pag --benches geneval \
  --run-tag "$TAG" --limits geneval=30 --keep-going

echo "########## PHASE 2: oneig x10 ##########"
python scripts/run_methods.py --model cosmos2 \
  --methods $ALL --benches oneig \
  --run-tag "$TAG" --limits oneig=15 --extra oneig:--grid=1x1 --keep-going

# PHASE 3 (dpg x10, ~900 imgs @ ~2.4min = ~36h) intentionally NOT run here:
# does not fit a single 12h V100 job. Handled via a dedicated run/scope.

echo "########## COSMOS2 geneval+oneig DONE ##########"
