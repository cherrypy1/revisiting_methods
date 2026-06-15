#!/bin/bash
# Setup script for MMDetection configs when installed from PyPI
# This downloads the necessary config files for Mask2Former

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
CONFIG_DIR="${PROJECT_ROOT}/configs"

echo "Setting up MMDetection configs in ${CONFIG_DIR}..."

# Create directory structure
mkdir -p "${CONFIG_DIR}/_base_/datasets"
mkdir -p "${CONFIG_DIR}/_base_/models"
mkdir -p "${CONFIG_DIR}/_base_/schedules"
mkdir -p "${CONFIG_DIR}/_base_/default_runtime"
mkdir -p "${CONFIG_DIR}/common/lsj_100e_coco_instance"
mkdir -p "${CONFIG_DIR}/mask2former"

MMDET_BASE_URL="https://raw.githubusercontent.com/open-mmlab/mmdetection/main/configs"

echo "Downloading base configs..."

# Base datasets
wget -q "${MMDET_BASE_URL}/_base_/datasets/coco_instance.py" \
    -O "${CONFIG_DIR}/_base_/datasets/coco_instance.py" || true

# Base default runtime  
wget -q "${MMDET_BASE_URL}/_base_/default_runtime.py" \
    -O "${CONFIG_DIR}/_base_/default_runtime.py" || true

echo "Downloading Mask2Former configs..."

# Main Mask2Former config
# Note: The config name changed in newer versions of mmdetection
wget -q "${MMDET_BASE_URL}/mask2former/mask2former_swin-s-p4-w7-224_8xb2-lsj-50e_coco.py" \
    -O "${CONFIG_DIR}/mask2former/mask2former_swin-s-p4-w7-224_lsj_8x2_50e_coco.py" 2>/dev/null || \
wget -q "${MMDET_BASE_URL}/mask2former/mask2former_swin-s-p4-w7-224_lsj_8x2_50e_coco.py" \
    -O "${CONFIG_DIR}/mask2former/mask2former_swin-s-p4-w7-224_lsj_8x2_50e_coco.py" 2>/dev/null || \
echo "Warning: Could not download Mask2Former config. You may need to install mmdet from source."

# Common LSJ config
wget -q "${MMDET_BASE_URL}/common/lsj_100e_coco_instance.py" \
    -O "${CONFIG_DIR}/common/lsj_100e_coco_instance.py" 2>/dev/null || true

echo "Done! Configs downloaded to ${CONFIG_DIR}"
echo ""
echo "If you encounter config issues, consider installing mmdetection from source:"
echo "  git clone https://github.com/open-mmlab/mmdetection.git"
echo "  cd mmdetection && pip install -v -e ."
