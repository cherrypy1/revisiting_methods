#!/usr/bin/env bash
# Install a separate GenEval evaluator environment.
#
# Upstream GenEval expects the mmdetection 2.x API. The main generation env may
# use mmdet 3.x, but that returns DetDataSample objects and breaks
# evaluation/evaluate_images.py. Keep this evaluator isolated.
#
# Usage:
#   bash scripts/install_geneval_eval.sh [geneval_root]
#
# Outputs:
#   ~/.venv_geneval_eval/bin/python
#   ~/geneval-bench/mmdetection-v2/

set -euo pipefail

GENEVAL_ROOT=${1:-${GENEVAL_ROOT:-$HOME/geneval-bench}}
BENCH_VENV=${BENCH_VENV:-$HOME/.venv_geneval_eval}
BENCH_PYTHON=$BENCH_VENV/bin/python
BOOTSTRAP_PYTHON=${BOOTSTRAP_PYTHON:-$HOME/_.venv/.venv/bin/python}

MMDET_DIR=${MMDET_DIR:-$GENEVAL_ROOT/mmdetection-v2}
MODELS_DIR=${GENEVAL_MODELS:-$GENEVAL_ROOT/models}
MM_CONFIG=${GENEVAL_MM_CONFIG:-$MMDET_DIR/configs/mask2former/mask2former_swin-s-p4-w7-224_lsj_8x2_50e_coco.py}
CKPT=$MODELS_DIR/mask2former_swin-s-p4-w7-224_lsj_8x2_50e_coco.pth
CKPT_URL=https://download.openmmlab.com/mmdetection/v2.0/mask2former/mask2former_swin-s-p4-w7-224_lsj_8x2_50e_coco/mask2former_swin-s-p4-w7-224_lsj_8x2_50e_coco_20220504_001756-743b7d99.pth

echo "== GenEval evaluator env =="

if [[ ! -x "$BOOTSTRAP_PYTHON" ]]; then
    BOOTSTRAP_PYTHON=python3
fi

if [[ ! -x "$BENCH_PYTHON" ]]; then
    echo "Creating venv -> $BENCH_VENV"
    "$BOOTSTRAP_PYTHON" -m venv "$BENCH_VENV"
fi

"$BENCH_PYTHON" -m pip install -U "pip<25" "setuptools<70" wheel

# CUDA/PyTorch stack compatible with mmcv-full 1.7.x wheels and V100 nodes.
"$BENCH_PYTHON" -m pip install \
    --extra-index-url https://download.pytorch.org/whl/cu117 \
    "torch==1.13.1+cu117" "torchvision==0.14.1+cu117"

# GenEval imports numpy/pandas/PIL/cv2 directly. Keep numpy below 2.x and use
# headless OpenCV so compute nodes do not need libGL.so.1.
"$BENCH_PYTHON" -m pip uninstall -y opencv-python opencv-python-headless || true
"$BENCH_PYTHON" -m pip install --only-binary=:all: \
    "numpy==1.23.5" \
    "pandas==1.5.3" \
    "Pillow==9.5.0" \
    "opencv-python-headless==4.8.1.78" \
    "pycocotools==2.0.7" \
    "scipy<1.12" \
    "matplotlib<3.8" \
    "PyYAML" \
    "tqdm" \
    "terminaltables" \
    "addict" \
    "yapf"

# OpenMMLab 2.x stack. Avoid openmim here: on the cluster it may pull pandas
# source builds. Install the exact prebuilt mmcv-full wheel directly instead.
"$BENCH_PYTHON" -m pip install \
    -f https://download.openmmlab.com/mmcv/dist/cu117/torch1.13/index.html \
    "mmcv-full==1.7.2"
"$BENCH_PYTHON" -m pip install "mmdet==2.28.2"

# CLIP color classifier used by GenEval.
"$BENCH_PYTHON" -m pip install \
    "open_clip_torch==2.24.0" \
    "clip-benchmark" \
    "sentencepiece==0.1.99" \
    "timm==0.9.16"

# Some packages above may try to move numpy/opencv. Clean interrupted pip
# uninstall leftovers (``-umpy`` / ``~umpy``) and restore ABI-safe pins without
# dependency resolution, so numpy is not upgraded back to 2.x.
SITE_PACKAGES=$("$BENCH_PYTHON" - <<'PY'
import site
print(site.getsitepackages()[0])
PY
)
rm -rf "$SITE_PACKAGES"/numpy "$SITE_PACKAGES"/numpy-*.dist-info \
       "$SITE_PACKAGES"/-umpy* "$SITE_PACKAGES"/~umpy*
"$BENCH_PYTHON" -m pip uninstall -y opencv-python || true
"$BENCH_PYTHON" -m pip install --force-reinstall --only-binary=:all: \
    --no-deps "numpy==1.23.5"
"$BENCH_PYTHON" -m pip install --force-reinstall --only-binary=:all: \
    --no-deps "pandas==1.5.3" "scipy==1.11.4" \
    "matplotlib==3.7.5" "opencv-python-headless==4.8.1.78"

if [[ ! -d "$MMDET_DIR" ]]; then
    echo "Cloning mmdetection v2.28.2 -> $MMDET_DIR"
    git clone --branch v2.28.2 --depth 1 https://github.com/open-mmlab/mmdetection.git "$MMDET_DIR"
else
    echo "mmdetection v2 checkout already exists: $MMDET_DIR"
fi

if [[ ! -f "$MM_CONFIG" ]]; then
    echo "!! Expected mmdet config missing: $MM_CONFIG" >&2
    exit 1
fi

mkdir -p "$MODELS_DIR"
if [[ ! -f "$CKPT" ]]; then
    echo "Downloading Mask2Former weights -> $CKPT"
    wget -q "$CKPT_URL" -O "$CKPT"
else
    echo "Mask2Former weights already present: $CKPT"
fi

echo
echo "== Verify imports =="
"$BENCH_PYTHON" - <<'PY'
import cv2
import numpy
import pandas
import torch
import mmcv
import mmdet
import open_clip
from clip_benchmark.metrics import zeroshot_classification as zsc

print("cv2", cv2.__version__)
print("numpy", numpy.__version__)
print("pandas", pandas.__version__)
print("torch", torch.__version__)
print("mmcv", mmcv.__version__)
print("mmdet", mmdet.__version__)
print("open_clip ok")
print("clip_benchmark ok")
PY

echo
echo "GenEval evaluator ready. Use:"
echo "  export PYTHONPATH=$HOME/inference"
echo "  export GENEVAL_MM_CONFIG=$MM_CONFIG"
echo "  $BENCH_PYTHON -m cfgbench.cli run configs/campaigns/cfg_geneval.yaml --stage validate"
