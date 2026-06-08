#!/usr/bin/env bash
# Install deps needed for GenEval evaluation (Mask2Former + CLIP).
#
# GenEval eval uses mmdet-3.x + a local clone of mmdetection (for the
# Mask2Former py-config) + mask2former swin-s coco weights.
#
# Usage: scripts/install_geneval_eval.sh [project_root]

set -euo pipefail

PROJECT_ROOT=${1:-${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}}

MMDET_DIR=${MMDET_DIR:-$PROJECT_ROOT/mmdetection}
MODELS_DIR=${GENEVAL_MODELS:-$PROJECT_ROOT/models}
MM_CONFIG=${GENEVAL_MM_CONFIG:-$MMDET_DIR/configs/mask2former/mask2former_swin-s-p4-w7-224_8xb2-lsj-50e_coco.py}
CKPT=$MODELS_DIR/mask2former_swin-s-p4-w7-224_lsj_8x2_50e_coco.pth
CKPT_URL=https://download.openmmlab.com/mmdetection/v2.0/mask2former/mask2former_swin-s-p4-w7-224_lsj_8x2_50e_coco/mask2former_swin-s-p4-w7-224_lsj_8x2_50e_coco_20220504_001756-743b7d99.pth

echo "== GenEval eval deps =="

# 1. Python packages (mmdet stack + clip deps). mmdet 3.x needs mmcv>=2 built
# against the exact torch version -> use openmim which handles CUDA builds.
pip install openmim
mim install "mmengine>=0.10" "mmcv>=2.0.0,<2.3" "mmdet>=3.3,<3.4"
pip install open_clip_torch clip-benchmark

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
echo "  export GENEVAL_MODELS=$MODELS_DIR"
echo "  export GENEVAL_MM_CONFIG=$MM_CONFIG"
