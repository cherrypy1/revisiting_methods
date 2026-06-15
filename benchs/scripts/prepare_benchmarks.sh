#!/usr/bin/env bash

set -euo pipefail

PROJECT_ROOT=${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
GENEVAL_ROOT=${GENEVAL_ROOT:-$PROJECT_ROOT/geneval-bench}
ONEIG_ROOT=${ONEIG_ROOT:-$PROJECT_ROOT/OneIG-Benchmark}
DPG_ROOT=${DPG_ROOT:-$PROJECT_ROOT/ELLA}
GENEVAL_CREATE_N=${GENEVAL_CREATE_N:-100}
GENEVAL_SMOKE_PER_TAG=${GENEVAL_SMOKE_PER_TAG:-1}
GENEVAL_EVALUATION_PER_TAG=${GENEVAL_EVALUATION_PER_TAG:-5}
ONEIG_SMOKE_PER_CATEGORY=${ONEIG_SMOKE_PER_CATEGORY:-1}
ONEIG_EVALUATION_PER_CATEGORY=${ONEIG_EVALUATION_PER_CATEGORY:-6}
DPG_EVALUATION_TOTAL=${DPG_EVALUATION_TOTAL:-30}
PROMPT_SEED=${PROMPT_SEED:-42}
PROMPT_ROOT=${PROMPT_ROOT:-$PROJECT_ROOT/prompts}

clone_if_missing() {
    local repo=$1
    local dst=$2
    if [[ -d "$dst/.git" ]]; then
        echo "[prepare] already exists: $dst"
    else
        echo "[prepare] cloning $repo -> $dst"
        git clone "$repo" "$dst"
    fi
}

link_file() {
    local src=$1
    local dst=$2
    mkdir -p "$(dirname "$dst")"
    rm -f "$dst"
    ln -s "$src" "$dst"
}

link_dir() {
    local src=$1
    local dst=$2
    mkdir -p "$(dirname "$dst")"
    rm -rf "$dst"
    ln -s "$src" "$dst"
}

cd "$PROJECT_ROOT"
mkdir -p "$PROMPT_ROOT"

clone_if_missing https://github.com/djghosh13/geneval.git "$GENEVAL_ROOT"
clone_if_missing https://github.com/OneIG-Bench/OneIG-Benchmark.git "$ONEIG_ROOT"
clone_if_missing https://github.com/TencentQQGYLab/ELLA.git "$DPG_ROOT"

GENEVAL_FULL=$GENEVAL_ROOT/prompts/evaluation_metadata.jsonl
if [[ ! -f "$GENEVAL_FULL" ]]; then
    echo "[prepare] creating full GenEval metadata"
    python scripts/prompts/create_prompts.py --seed "$PROMPT_SEED" -n "$GENEVAL_CREATE_N" -o "$GENEVAL_ROOT/prompts"
fi

GENEVAL_SMOKE=$PROMPT_ROOT/smoke_test/geneval.jsonl
GENEVAL_EVALUATION=$PROMPT_ROOT/evaluation/geneval.jsonl
GENEVAL_FULL_LINK=$PROMPT_ROOT/full_test/geneval.jsonl
python scripts/prompts/prepare_geneval_prompts.py \
    --input "$GENEVAL_FULL" \
    --output "$GENEVAL_SMOKE" \
    --per-tag "$GENEVAL_SMOKE_PER_TAG" \
    --seed "$PROMPT_SEED"
python scripts/prompts/prepare_geneval_prompts.py \
    --input "$GENEVAL_FULL" \
    --output "$GENEVAL_EVALUATION" \
    --per-tag "$GENEVAL_EVALUATION_PER_TAG" \
    --seed "$PROMPT_SEED"
link_file "$GENEVAL_FULL" "$GENEVAL_FULL_LINK"

ONEIG_CSV_FULL=$ONEIG_ROOT/OneIG-Bench.csv
ONEIG_SMOKE=$PROMPT_ROOT/smoke_test/oneig.csv
ONEIG_EVALUATION=$PROMPT_ROOT/evaluation/oneig.csv
ONEIG_FULL_LINK=$PROMPT_ROOT/full_test/oneig.csv
python scripts/prompts/prepare_oneig_prompts.py \
    --input-csv "$ONEIG_CSV_FULL" \
    --output-csv "$ONEIG_SMOKE" \
    --per-category "$ONEIG_SMOKE_PER_CATEGORY" \
    --seed "$PROMPT_SEED"
python scripts/prompts/prepare_oneig_prompts.py \
    --input-csv "$ONEIG_CSV_FULL" \
    --output-csv "$ONEIG_EVALUATION" \
    --per-category "$ONEIG_EVALUATION_PER_CATEGORY" \
    --seed "$PROMPT_SEED"
link_file "$ONEIG_CSV_FULL" "$ONEIG_FULL_LINK"

DPG_CSV_FULL=${DPG_CSV_FULL:-}
if [[ -z "$DPG_CSV_FULL" ]]; then
    DPG_CSV_FULL=$(find "$DPG_ROOT/dpg_bench" -maxdepth 2 -type f -name "*dpg*.csv" | head -n 1)
fi
if [[ -z "$DPG_CSV_FULL" || ! -f "$DPG_CSV_FULL" ]]; then
    echo "[prepare] DPG csv not found under $DPG_ROOT/dpg_bench" >&2
    exit 1
fi

DPG_SMOKE_ROOT=$PROMPT_ROOT/smoke_test/dpg
DPG_EVALUATION_ROOT=$PROMPT_ROOT/evaluation/dpg
DPG_FULL_ROOT=$PROMPT_ROOT/full_test/dpg
python scripts/prompts/prepare_dpg_prompts.py \
    --input-csv "$DPG_CSV_FULL" \
    --input-prompts-dir "$DPG_ROOT/dpg_bench/prompts" \
    --output-csv "$DPG_SMOKE_ROOT/dpg_bench.csv" \
    --output-prompts-dir "$DPG_SMOKE_ROOT/prompts" \
    --per-category 1 \
    --seed "$PROMPT_SEED"
python scripts/prompts/prepare_dpg_prompts.py \
    --input-csv "$DPG_CSV_FULL" \
    --input-prompts-dir "$DPG_ROOT/dpg_bench/prompts" \
    --output-csv "$DPG_EVALUATION_ROOT/dpg_bench.csv" \
    --output-prompts-dir "$DPG_EVALUATION_ROOT/prompts" \
    --total "$DPG_EVALUATION_TOTAL" \
    --seed "$PROMPT_SEED"
link_file "$DPG_CSV_FULL" "$DPG_FULL_ROOT/dpg_bench.csv"
link_dir "$DPG_ROOT/dpg_bench/prompts" "$DPG_FULL_ROOT/prompts"

if [[ "${INSTALL_EVAL:-0}" == "1" ]]; then
    bash scripts/setup/install_geneval_eval.sh "$GENEVAL_ROOT"
    bash scripts/setup/install_dpg_eval.sh "$DPG_ROOT"
fi

if [[ "${INSTALL_ONEIG_EVAL:-0}" == "1" ]]; then
    bash scripts/setup/install_oneig_eval.sh "$ONEIG_ROOT"
fi

echo
echo "[prepare] done"
echo "export GENEVAL_ROOT=$GENEVAL_ROOT"
echo "export ONEIG_ROOT=$ONEIG_ROOT"
echo "export DPG_ROOT=$DPG_ROOT"
echo "prompt sets:"
echo "  smoke_test: $PROMPT_ROOT/smoke_test"
echo "  evaluation: $PROMPT_ROOT/evaluation"
echo "  full_test: $PROMPT_ROOT/full_test"
