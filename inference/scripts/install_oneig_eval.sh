#!/usr/bin/env bash
# Install deps needed to run OneIG-Benchmark evaluation on top of an existing
# SD3.5 generation env. Clones the benchmark repo under $ONEIG_ROOT (default
# $HOME/OneIG-Benchmark) and pip-installs its extra requirements.
#
# Usage: scripts/install_oneig_eval.sh [oneig_root]

set -euo pipefail

ONEIG_ROOT=${1:-${ONEIG_ROOT:-$HOME/OneIG-Benchmark}}
ONEIG_REPO=https://github.com/OneIG-Bench/OneIG-Benchmark.git

if [[ ! -d "$ONEIG_ROOT" ]]; then
    echo "Cloning OneIG-Benchmark → $ONEIG_ROOT"
    git clone "$ONEIG_REPO" "$ONEIG_ROOT"
else
    echo "OneIG-Benchmark already at $ONEIG_ROOT — skipping clone"
fi

# OneIG eval pins transformers==4.50.0 (newer breaks mPLUG-Owl3 loader).
# Rest of the pins live in OneIG-Benchmark/requirements.txt.
pip install "transformers==4.50.0"
if [[ -f "$ONEIG_ROOT/requirements.txt" ]]; then
    pip install -r "$ONEIG_ROOT/requirements.txt"
fi

echo "OneIG eval deps installed. Export:"
echo "  export ONEIG_ROOT=$ONEIG_ROOT"
echo "  export ONEIG_CSV=$ONEIG_ROOT/OneIG-Bench.csv"
