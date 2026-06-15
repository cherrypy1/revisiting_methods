#!/usr/bin/env bash
# Cosmos2 DPG image generation (run on an allocated compute node).
# Full 90-prompt parity with SD3.5; ~900 imgs @ ~3.7min = ~55h total, so this
# spans several jobs. generate_dpg.py skips existing {idx}.png, so re-running on
# each new allocation resumes where the last left off.
# Eval (mplug VQA) is deferred — run once all 10 methods' images exist.
set -u
cd ~/geneval
module purge
module load gnu14/14.1
source ~/.venv/bin/activate

TAG=cosmos2_24052026
# Methods can be overridden via CLI so multiple nodes take DISJOINT subsets
# (running the same full list on two nodes would race on identical {idx}.png).
#   bash _launch_cosmos2_dpg.sh cfg no_cfg cfgpp cfg0s apg     # node A
#   bash _launch_cosmos2_dpg.sh tcfg sag seg_sigma10 oseg pag  # node B
if [ "$#" -gt 0 ]; then
  METHODS="$*"
else
  METHODS="cfg no_cfg cfgpp cfg0s apg tcfg sag seg_sigma10 oseg pag"
fi

echo "########## COSMOS2 DPG gen (resumable, skip-eval) :: $METHODS ##########"
python scripts/run_methods.py --model cosmos2 \
  --methods $METHODS --benches dpg \
  --run-tag "$TAG" --limits dpg=90 \
  --extra dpg:"--pic-num=1 --skip-eval" --keep-going

echo "########## COSMOS2 DPG gen pass DONE ##########"
