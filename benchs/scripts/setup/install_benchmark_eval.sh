#!/usr/bin/env bash
# Install all benchmark evaluator dependencies into one isolated venv.
#
# Generation still uses the main environment. External benchmark scorers run
# from $BENCH_VENV so their pins cannot break diffusers/Flux.

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=$(cd "$SCRIPT_DIR/../.." && pwd)
BENCH_VENV=${BENCH_VENV:-$PROJECT_ROOT/.venv_bench}

export BENCH_VENV

bash "$SCRIPT_DIR/install_geneval_eval.sh"
bash "$SCRIPT_DIR/install_dpg_eval.sh"
bash "$SCRIPT_DIR/install_oneig_eval.sh"

echo
echo "All benchmark evaluator deps installed into:"
echo "  $BENCH_VENV"
echo "Export if non-default:"
echo "  export BENCH_PYTHON=$BENCH_VENV/bin/python"
