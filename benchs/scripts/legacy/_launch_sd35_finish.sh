#!/usr/bin/env bash
# SD3.5 finish (run on the allocated compute node).
# Phase 1: smoke the consolidated pipeline — 2 imgs x 10 methods, geneval, no eval.
# Phase 2: PAG dpg eval on already-generated images (soundfile stub now fixed).
set -u
cd ~/geneval
module purge
module load gnu14/14.1
source ~/.venv/bin/activate

echo "########## PHASE 1: consolidated smoke (10 methods x 2 imgs) ##########"
python scripts/run_methods.py \
  --methods no_cfg cfg cfgpp cfg0s apg tcfg sag seg_sigma10_cfg3 oseg pag \
  --benches geneval --run-tag smoke_consolidated_02062026 \
  --limits geneval=2 --extra geneval:--skip-eval --keep-going

echo "########## PHASE 2: PAG dpg eval (existing images) ##########"
python scripts/eval/eval_dpg.py --dpg-root ~/ELLA \
  --image-dir ~/geneval/outputs/dpg/pag_strat27042026/images \
  --pic-num 1 --resolution 1024 \
  --out-dir ~/geneval/outputs/dpg/pag_strat27042026/eval

echo "########## SD3.5 FINISH DONE ##########"
