#!/usr/bin/env bash
# Smoke test: 8 guidance methods × 3 benchmarks × ~25 images per method.
# Skips evaluation stage to keep it fast. Verifies generation end-to-end.
#
# Usage:
#   bash scripts/smoke.sh                 # all benches, all 8 methods
#   bash scripts/smoke.sh geneval         # single bench
#   bash scripts/smoke.sh geneval cfg     # single bench, single method

set -euo pipefail

PROJECT_ROOT=${PROJECT_ROOT:-/home/aaturevich/geneval}
VENV=${VENV:-/home/aaturevich/.venv}
RUN_TAG=${RUN_TAG:-smoke_$(date +%d%m%Y_%H%M)}
OUT_ROOT=${OUT_ROOT:-"$PROJECT_ROOT/outputs"}
LIMIT=${LIMIT:-25}
N_SAMPLES=${N_SAMPLES:-1}
# OneIG stores 2x2 grids (4 internal samples each). Keep its prompt count low
# so total generated images roughly match LIMIT per method per bench.
ONEIG_PER_CAT=${ONEIG_PER_CAT:-2}

METHODS=(cfg seg_sigma10_cfg3 pag sag oseg tcfg cfgpp cfg0s)
BENCHES=(geneval oneig dpg)

if [[ $# -ge 1 ]]; then
    BENCHES=("$1")
fi
if [[ $# -ge 2 ]]; then
    METHODS=("$2")
fi

# shellcheck disable=SC1091
source "$VENV/bin/activate"
cd "$PROJECT_ROOT"

smoke_geneval() {
    local method=$1
    local config="$PROJECT_ROOT/generation/configs/sd35/${method}.py"
    local prompts=${GENEVAL_PROMPTS:-"$PROJECT_ROOT/prompts/evaluation_metadata.jsonl"}
    local out="$OUT_ROOT/${RUN_TAG}/geneval/${method}"
    mkdir -p "$out"
    python generation/diffusers_generate.py "$prompts" \
        --config "$config" --outdir "$out" \
        --limit "$LIMIT" --n_samples "$N_SAMPLES" --skip_grid
}

smoke_oneig() {
    local method=$1
    local config="$PROJECT_ROOT/generation/configs/sd35/${method}.py"
    local csv=${ONEIG_CSV:-/home/aaturevich/OneIG-Benchmark/OneIG-Bench.csv}
    local out="$OUT_ROOT/${RUN_TAG}/oneig"
    mkdir -p "$out"
    python scripts/generate_oneig.py "$config" \
        --csv "$csv" --out-dir "$out" --model-name "$method" \
        --limit "$ONEIG_PER_CAT"
}

smoke_dpg() {
    local method=$1
    local config="$PROJECT_ROOT/generation/configs/sd35/${method}.py"
    local prompts_dir=${DPG_PROMPTS:-/home/aaturevich/ELLA/dpg_bench/prompts}
    local out="$OUT_ROOT/${RUN_TAG}/dpg/${method}/images"
    mkdir -p "$out"
    python scripts/generate_dpg.py "$config" \
        --prompts-dir "$prompts_dir" --out-dir "$out" \
        --pic-num 1 --resolution 1024 --limit "$LIMIT"
}

for bench in "${BENCHES[@]}"; do
    for method in "${METHODS[@]}"; do
        echo "=== smoke: $method / $bench ==="
        case "$bench" in
            geneval) smoke_geneval "$method" ;;
            oneig)   smoke_oneig "$method" ;;
            dpg)     smoke_dpg "$method" ;;
            *) echo "unknown bench: $bench" >&2; exit 2 ;;
        esac
    done
done

echo "Smoke outputs: $OUT_ROOT/${RUN_TAG}"
