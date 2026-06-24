#!/usr/bin/env bash
# Re-run all benches for PAG only, with fixed L13 config.
set -euo pipefail
cd "$HOME/geneval"
source /etc/profile.d/modules.sh 2>/dev/null || true
module purge 2>/dev/null || true
module load gnu14/14.1 2>/dev/null || true
source "$HOME/.venv/bin/activate"
TAG=strat27042026
M=pag

# Note: do NOT wipe — generation skips existing imgs.

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "##### geneval #####"
python scripts/bench.py "$M" geneval --limit 90 --run-tag "$TAG"
echo "##### oneig #####"
python scripts/bench.py "$M" oneig  --limit 15 --grid 1x1 --run-tag "$TAG"
echo "##### dpg #####"
python scripts/bench.py "$M" dpg    --limit 90 --pic-num 1 --run-tag "$TAG"
echo "ALL DONE"
