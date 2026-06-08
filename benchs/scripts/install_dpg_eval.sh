#!/usr/bin/env bash
# Install deps needed to run DPG-Bench evaluation (via ELLA/dpg_bench).
# Clones ELLA under $DPG_ROOT (default $HOME/ELLA) and pip-installs extras.
#
# Usage: scripts/install_dpg_eval.sh [dpg_root]

set -euo pipefail

DPG_ROOT=${1:-${DPG_ROOT:-$HOME/ELLA}}
ELLA_REPO=https://github.com/TencentQQGYLab/ELLA.git

if [[ ! -d "$DPG_ROOT" ]]; then
    echo "Cloning ELLA → $DPG_ROOT"
    git clone "$ELLA_REPO" "$DPG_ROOT"
else
    echo "ELLA already at $DPG_ROOT — skipping clone"
fi

# DPG-Bench scoring uses mPLUG-Owl-large as VQA model. ELLA's README lists
# the extra pip pins; re-run here if an explicit requirements file exists.
if [[ -f "$DPG_ROOT/dpg_bench/requirements.txt" ]]; then
    pip install -r "$DPG_ROOT/dpg_bench/requirements.txt"
fi

echo "DPG eval deps installed. Export:"
echo "  export DPG_ROOT=$DPG_ROOT"
echo "  export DPG_PROMPTS=$DPG_ROOT/dpg_bench/prompts"
