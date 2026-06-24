#!/usr/bin/env bash
# Run every method config on a single benchmark.
#
# Usage:
#   scripts/run_all_methods.sh <benchmark> [method1 method2 ...]
#
# Defaults to all methods under generation/configs/sd35/ when none given.

set -euo pipefail

BENCH=${1:?"benchmark required"}
shift || true

PROJECT_ROOT=${PROJECT_ROOT:-/home/aaturevich/geneval}
CONFIG_DIR="$PROJECT_ROOT/generation/configs/sd35"

if [[ $# -gt 0 ]]; then
    METHODS=("$@")
else
    mapfile -t METHODS < <(
        find "$CONFIG_DIR" -maxdepth 1 -name '*.py' ! -name '_*.py' \
        -printf '%f\n' | sed 's/\.py$//' | sort
    )
fi

for method in "${METHODS[@]}"; do
    echo "=== $method on $BENCH ==="
    bash "$(dirname "$0")/bench.sh" "$method" "$BENCH"
done
