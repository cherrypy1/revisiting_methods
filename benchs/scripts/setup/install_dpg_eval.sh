#!/usr/bin/env bash
# Install deps needed to run DPG-Bench evaluation (via ELLA/dpg_bench).
# Clones ELLA under $DPG_ROOT (default $PROJECT_ROOT/ELLA) and installs extras
# into $BENCH_VENV, not the main generation environment.
#
# Usage: scripts/setup/install_dpg_eval.sh [dpg_root]

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=$(cd "$SCRIPT_DIR/../.." && pwd)
DPG_ROOT=${1:-${DPG_ROOT:-$PROJECT_ROOT/ELLA}}
BENCH_VENV=${BENCH_VENV:-$PROJECT_ROOT/.venv_bench}
BENCH_PYTHON=${BENCH_PYTHON:-$BENCH_VENV/bin/python}
ELLA_REPO=https://github.com/TencentQQGYLab/ELLA.git

if [[ ! -d "$DPG_ROOT" ]]; then
    echo "Cloning ELLA → $DPG_ROOT"
    git clone "$ELLA_REPO" "$DPG_ROOT"
else
    echo "ELLA already at $DPG_ROOT — skipping clone"
fi

if [[ ! -x "$BENCH_PYTHON" ]]; then
    echo "Creating benchmark evaluator venv -> $BENCH_VENV"
    python -m venv "$BENCH_VENV"
fi

"$BENCH_PYTHON" -m pip install -U pip setuptools wheel
"$BENCH_PYTHON" -m pip install \
    --extra-index-url https://download.pytorch.org/whl/cu121 \
    "torch==2.5.1+cu121" "torchvision==0.20.1+cu121"
"$BENCH_PYTHON" -m pip install accelerate

# DPG-Bench scoring uses mPLUG-Owl-large as VQA model. ELLA's README lists
# the extra pip pins; re-run here if an explicit requirements file exists.
if [[ -f "$DPG_ROOT/dpg_bench/requirements.txt" ]]; then
    "$BENCH_PYTHON" -m pip install -r "$DPG_ROOT/dpg_bench/requirements.txt"
fi

echo "DPG eval deps installed. Export:"
echo "  export BENCH_PYTHON=$BENCH_PYTHON"
echo "  export DPG_ROOT=$DPG_ROOT"
echo "  export DPG_PROMPTS=$DPG_ROOT/dpg_bench/prompts"
