#!/usr/bin/env bash
# Install deps needed for GenEval evaluation (Mask2Former + CLIP).
#
# GenEval eval uses mmdet-3.x + a local clone of mmdetection (for the
# Mask2Former py-config) + mask2former swin-s coco weights. Python deps are
# installed into $BENCH_VENV, not the main generation environment.
#
# Usage: scripts/setup/install_geneval_eval.sh [geneval_root]

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
CFG_ROOT=$(cd "$SCRIPT_DIR/../.." && pwd)
GENEVAL_ROOT=${1:-${GENEVAL_ROOT:-$CFG_ROOT/geneval-bench}}
BENCH_VENV=${BENCH_VENV:-$CFG_ROOT/.venv_bench}
BENCH_PYTHON=${BENCH_PYTHON:-$BENCH_VENV/bin/python}
MIM_BIN=$BENCH_VENV/bin/mim

MMDET_DIR=${MMDET_DIR:-$GENEVAL_ROOT/mmdetection}
MODELS_DIR=${GENEVAL_MODELS:-$GENEVAL_ROOT/models}
MM_CONFIG=${GENEVAL_MM_CONFIG:-$MMDET_DIR/configs/mask2former/mask2former_swin-s-p4-w7-224_8xb2-lsj-50e_coco.py}
CKPT=$MODELS_DIR/mask2former_swin-s-p4-w7-224_lsj_8x2_50e_coco.pth
CKPT_URL=https://download.openmmlab.com/mmdetection/v2.0/mask2former/mask2former_swin-s-p4-w7-224_lsj_8x2_50e_coco/mask2former_swin-s-p4-w7-224_lsj_8x2_50e_coco_20220504_001756-743b7d99.pth

echo "== GenEval eval deps =="

if [[ ! -x "$BENCH_PYTHON" ]]; then
    echo "Creating benchmark evaluator venv -> $BENCH_VENV"
    python -m venv "$BENCH_VENV"
fi

"$BENCH_PYTHON" -m pip install -U pip setuptools wheel

# mmdet/mmcv need torch present before openmim resolves CUDA wheels.
"$BENCH_PYTHON" -m pip install \
    --extra-index-url https://download.pytorch.org/whl/cu121 \
    "torch==2.5.1+cu121" "torchvision==0.20.1+cu121"

# Python packages (mmdet stack + clip deps). mmdet 3.x needs mmcv>=2 built
# against the exact torch version -> use openmim inside the evaluator venv.
"$BENCH_PYTHON" -m pip install openmim
"$MIM_BIN" install "mmengine>=0.10" "mmcv>=2.0.0,<2.3" "mmdet>=3.3,<3.4"
"$BENCH_PYTHON" -m pip install open_clip_torch clip-benchmark

# 2. mmdetection repo (configs only, no install)
if [[ ! -d "$MMDET_DIR" ]]; then
    echo "Cloning mmdetection → $MMDET_DIR"
    git clone --depth 1 https://github.com/open-mmlab/mmdetection.git "$MMDET_DIR"
else
    echo "mmdetection already at $MMDET_DIR — skipping clone"
fi

if [[ ! -f "$MM_CONFIG" ]]; then
    echo "!! Expected mmdet config missing: $MM_CONFIG" >&2
    echo "   Check mmdetection repo or override GENEVAL_MM_CONFIG." >&2
fi

# 3. Mask2Former weights
mkdir -p "$MODELS_DIR"
if [[ ! -f "$CKPT" ]]; then
    echo "Downloading Mask2Former weights → $CKPT"
    wget -q "$CKPT_URL" -O "$CKPT"
else
    echo "Mask2Former weights already present: $CKPT"
fi

echo
echo "GenEval eval deps installed. Export if non-default:"
echo "  export BENCH_PYTHON=$BENCH_PYTHON"
echo "  export GENEVAL_MODELS=$MODELS_DIR"
echo "  export GENEVAL_MM_CONFIG=$MM_CONFIG"
