#!/usr/bin/env bash
# Cosmos2 round-2 worker: finishes all remaining geneval/dpg/oneig gaps.
# Usage on each allocated compute node:  bash -l scripts/_launch_cosmos2_round2.sh <slot 0..3>
#
# Work is partitioned into 4 balanced slots (~230 imgs each, ~14h @ V100).
# Method subsets are DISJOINT within every bench so two nodes never race on the
# same {idx}.png. All generators skip existing files => fully resumable; if a
# 12h job dies mid-slot, re-running the same slot on a fresh job continues it.
#
#   slot0: geneval[seg_sigma10 pag]  dpg[tcfg]        oneig[cfg no_cfg]
#   slot1:                           dpg[sag oseg]    oneig[cfgpp]
#   slot2:                           dpg[apg pag]     oneig[cfg0s apg tcfg]
#   slot3:                           dpg[cfg0s]       oneig[sag seg_sigma10 oseg pag]
set -u
cd ~/geneval
module purge
module load gnu14/14.1
source ~/.venv/bin/activate

TAG=cosmos2_24052026
SLOT="${1:?usage: $0 <slot 0..3>}"

run_geneval() { python scripts/run_methods.py --model cosmos2 --methods "$@" \
  --benches geneval --run-tag "$TAG" --limits geneval=30 --keep-going; }
run_dpg() { python scripts/run_methods.py --model cosmos2 --methods "$@" \
  --benches dpg --run-tag "$TAG" --limits dpg=90 --extra dpg:"--pic-num=1 --skip-eval" --keep-going; }
run_oneig() { python scripts/run_methods.py --model cosmos2 --methods "$@" \
  --benches oneig --run-tag "$TAG" --limits oneig=15 --extra oneig:--grid=1x1 --keep-going; }

echo "########## COSMOS2 round2 slot=$SLOT ##########"
case "$SLOT" in
  0) run_geneval seg_sigma10 pag; run_dpg tcfg;      run_oneig cfg no_cfg ;;
  1) run_dpg sag oseg;                                run_oneig cfgpp ;;
  2) run_dpg apg pag;                                 run_oneig cfg0s apg tcfg ;;
  3) run_dpg cfg0s;                                   run_oneig sag seg_sigma10 oseg pag ;;
  *) echo "bad slot $SLOT (need 0..3)"; exit 1 ;;
esac
echo "########## COSMOS2 round2 slot=$SLOT DONE ##########"
