#!/usr/bin/env bash

set -euo pipefail

usage() {
    cat <<'USAGE'
Usage:
  bash scripts/smoke_test.sh <model> <bench...> <method...> [-- extra bench.py args]

Examples:
  bash scripts/smoke_test.sh flux2_klein_base all cfg -- --skip-eval
  bash scripts/smoke_test.sh sd35 geneval dpg cfg pag -- --skip-eval
USAGE
}

parse_and_exec() {
    local mode=$1
    shift

    if [[ $# -lt 3 ]]; then
        usage >&2
        exit 2
    fi

    local script_dir
    script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
    local model=$1
    shift

    local benches=()
    while [[ $# -gt 0 ]]; do
        case "$1" in
            geneval|oneig|dpg|all)
                benches+=("$1")
                shift
                ;;
            *)
                break
                ;;
        esac
    done

    local methods=()
    local extra=()
    while [[ $# -gt 0 ]]; do
        if [[ "$1" == "--" ]]; then
            shift
            extra=("$@")
            break
        fi
        methods+=("$1")
        shift
    done

    if [[ ${#benches[@]} -eq 0 || ${#methods[@]} -eq 0 ]]; then
        usage >&2
        exit 2
    fi

    local cmd=(
        python "$script_dir/evaluate.py"
        --mode "$mode"
        --model "$model"
        --benches "${benches[@]}"
        --methods "${methods[@]}"
    )
    if [[ -n "${RUN_TAG:-}" ]]; then
        cmd+=(--run-tag "$RUN_TAG")
    fi
    if [[ "${DRY_RUN:-0}" == "1" ]]; then
        cmd+=(--dry-run)
    fi
    if [[ "${KEEP_GOING:-0}" == "1" ]]; then
        cmd+=(--keep-going)
    fi
    if [[ ${#extra[@]} -gt 0 ]]; then
        cmd+=(-- "${extra[@]}")
    fi

    exec "${cmd[@]}"
}

parse_and_exec smoke_test "$@"
