#!/usr/bin/env bash
# Unified benchmark driver for SD3.5 guidance methods.
#
# Usage:
#   scripts/bench.sh <method> <benchmark>
#
# <method>     Name of a config under generation/configs/sd35/ (without .py).
# <benchmark>  geneval | oneig | dpg
#
# Env overrides (with remote defaults):
#   PROJECT_ROOT  /home/aaturevich/geneval
#   VENV          /home/aaturevich/.venv
#   RUN_TAG       $(date +%d%m%Y)
#   GENEVAL_PROMPTS   "$PROJECT_ROOT/prompts/evaluation_metadata.jsonl"
#   GENEVAL_MODELS    "$PROJECT_ROOT/models"
#   GENEVAL_MM_CONFIG "$PROJECT_ROOT/mmdetection/configs/mask2former/mask2former_swin-s-p4-w7-224_8xb2-lsj-50e_coco.py"
#   ONEIG_CSV   /home/aaturevich/OneIG-Benchmark/OneIG-Bench.csv
#   ONEIG_ROOT  /home/aaturevich/OneIG-Benchmark
#   DPG_ROOT    /home/aaturevich/ELLA
#   OUT_ROOT    "$PROJECT_ROOT/outputs"

set -euo pipefail

METHOD=${1:?"method name required (see generation/configs/sd35/)"}
BENCH=${2:?"benchmark required: geneval|oneig|dpg"}

PROJECT_ROOT=${PROJECT_ROOT:-/home/aaturevich/geneval}
VENV=${VENV:-/home/aaturevich/.venv}
RUN_TAG=${RUN_TAG:-$(date +%d%m%Y)}
OUT_ROOT=${OUT_ROOT:-"$PROJECT_ROOT/outputs"}
CONFIG="$PROJECT_ROOT/generation/configs/sd35/${METHOD}.py"

if [[ ! -f "$CONFIG" ]]; then
    echo "Config not found: $CONFIG" >&2
    exit 2
fi

# shellcheck disable=SC1091
source "$VENV/bin/activate"
cd "$PROJECT_ROOT"

run_geneval() {
    local prompts=${GENEVAL_PROMPTS:-"$PROJECT_ROOT/prompts/evaluation_metadata.jsonl"}
    local models=${GENEVAL_MODELS:-"$PROJECT_ROOT/models"}
    local mm_config=${GENEVAL_MM_CONFIG:-"$PROJECT_ROOT/mmdetection/configs/mask2former/mask2former_swin-s-p4-w7-224_8xb2-lsj-50e_coco.py"}
    local out="$OUT_ROOT/geneval/${METHOD}_${RUN_TAG}"
    mkdir -p "$out"
    python generation/diffusers_generate.py "$prompts" --config "$CONFIG" --outdir "$out"
    python evaluation/evaluate_images.py "$out" \
        --outfile "$out/results.jsonl" \
        --model-path "$models" \
        --model-config "$mm_config"
    python evaluation/summary_scores.py "$out/results.jsonl" | tee "$out/summary.txt"
}

run_oneig() {
    local csv=${ONEIG_CSV:-/home/aaturevich/OneIG-Benchmark/OneIG-Bench.csv}
    local out="$OUT_ROOT/oneig/${RUN_TAG}"
    mkdir -p "$out"
    python scripts/generate_oneig.py "$CONFIG" \
        --csv "$csv" --out-dir "$out" --model-name "$METHOD"
    echo "Generation done. Run OneIG evaluation from $ONEIG_ROOT:" >&2
    echo "  IMAGE_DIR=$out bash run_overall.sh  # after editing MODEL_NAMES" >&2
}

run_dpg() {
    local prompts_dir=${DPG_PROMPTS:-/home/aaturevich/ELLA/dpg_bench/prompts}
    local out="$OUT_ROOT/dpg/${METHOD}_${RUN_TAG}/images"
    mkdir -p "$out"
    python scripts/generate_dpg.py "$CONFIG" \
        --prompts-dir "$prompts_dir" --out-dir "$out" --pic-num 4 --resolution 1024
    echo "Generation done. To evaluate:" >&2
    echo "  cd ${DPG_ROOT:-/home/aaturevich/ELLA} && bash dpg_bench/dist_eval.sh $(dirname "$out") 1024" >&2
}

case "$BENCH" in
    geneval) run_geneval ;;
    oneig)   run_oneig ;;
    dpg)     run_dpg ;;
    *) echo "unknown benchmark: $BENCH" >&2; exit 2 ;;
esac
