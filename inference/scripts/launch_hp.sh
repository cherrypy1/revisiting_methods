#!/bin/bash
# Wrapper: module load gnu14 + run HP sweep on a single GPU with given methods.
# Usage: launch_hp.sh GPU_IDX TAG METHODS...
set -u
GPU="$1"; shift
TAG="$1"; shift
METHODS=("$@")
cd "$HOME/geneval"
mkdir -p logs
source /etc/profile.d/modules.sh 2>/dev/null || true
module purge 2>/dev/null || true
module load gnu14/14.1 2>/dev/null || true
LOG="logs/hp_${TAG}_gpu${GPU}.log"
echo "[launch] GPU=$GPU TAG=$TAG methods=${METHODS[*]} -> $LOG"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
CUDA_VISIBLE_DEVICES="$GPU" exec "$HOME/.venv/bin/python" scripts/cosmos_hp_run.py \
    --methods "${METHODS[@]}" --per-tag 2 --steps 20 > "$LOG" 2>&1
