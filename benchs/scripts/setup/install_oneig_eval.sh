#!/usr/bin/env bash
# Install deps needed to run OneIG-Benchmark evaluation on top of an existing
# SD3.5/Flux generation env. Clones the benchmark repo under $ONEIG_ROOT
# (default $PROJECT_ROOT/OneIG-Benchmark) and installs evaluator requirements
# into $BENCH_VENV so OneIG's transformers pin cannot break generation.
#
# Usage: scripts/setup/install_oneig_eval.sh [oneig_root]

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=$(cd "$SCRIPT_DIR/../.." && pwd)
ONEIG_ROOT=${1:-${ONEIG_ROOT:-$PROJECT_ROOT/OneIG-Benchmark}}
BENCH_VENV=${BENCH_VENV:-$PROJECT_ROOT/.venv_bench}
BENCH_PYTHON=${BENCH_PYTHON:-$BENCH_VENV/bin/python}
ONEIG_PYTHON=${ONEIG_PYTHON:-$BENCH_PYTHON}
ONEIG_REPO=https://github.com/OneIG-Bench/OneIG-Benchmark.git

if [[ ! -d "$ONEIG_ROOT" ]]; then
    echo "Cloning OneIG-Benchmark → $ONEIG_ROOT"
    git clone "$ONEIG_REPO" "$ONEIG_ROOT"
else
    echo "OneIG-Benchmark already at $ONEIG_ROOT — skipping clone"
fi

if [[ ! -x "$ONEIG_PYTHON" ]]; then
    echo "Creating benchmark evaluator venv -> $BENCH_VENV"
    python -m venv "$BENCH_VENV"
fi

"$ONEIG_PYTHON" -m pip install -U pip setuptools wheel

# OneIG evaluation uses torch-backed VLM scorers. Keep torch in this isolated
# env too, but do not change the main generation env.
"$ONEIG_PYTHON" -m pip install \
    --extra-index-url https://download.pytorch.org/whl/cu121 \
    "torch==2.5.1+cu121" "torchvision==0.20.1+cu121"

# OneIG eval pins transformers==4.50.0 (newer breaks mPLUG-Owl3 loader).
# Rest of the pins live in OneIG-Benchmark/requirements.txt.
"$ONEIG_PYTHON" -m pip install "transformers==4.50.0"
if [[ -f "$ONEIG_ROOT/requirements.txt" ]]; then
    "$ONEIG_PYTHON" -m pip install -r "$ONEIG_ROOT/requirements.txt"
fi

echo "OneIG eval deps installed. Export:"
echo "  export BENCH_PYTHON=$ONEIG_PYTHON"
echo "  export ONEIG_ROOT=$ONEIG_ROOT"
echo "  export ONEIG_CSV=$ONEIG_ROOT/OneIG-Bench.csv"
echo "  export ONEIG_PYTHON=$ONEIG_PYTHON"
