#!/bin/bash
# Full installation script for GenEval
# Usage: ./scripts/install.sh [cuda_version]
# Example: ./scripts/install.sh cu118
#          ./scripts/install.sh cu121
#          ./scripts/install.sh cpu

set -e

CUDA_VERSION="${1:-cu118}"
TORCH_VERSION="2.1"
MMCV_VERSION="2.1.0"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

echo "=============================================="
echo "GenEval Installation Script"
echo "CUDA version: ${CUDA_VERSION}"
echo "=============================================="

# Check if we're in a virtual environment
if [ -z "$VIRTUAL_ENV" ]; then
    echo "Warning: No virtual environment detected."
    echo "It's recommended to create one first:"
    echo "  python -m venv venv"
    echo "  source venv/bin/activate"
    read -p "Continue anyway? [y/N] " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

echo ""
echo "Step 1: Upgrading pip..."
pip install --upgrade pip setuptools wheel

echo ""
echo "Step 2: Installing PyTorch..."
if [ "$CUDA_VERSION" = "cpu" ]; then
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
else
    pip install torch torchvision --index-url "https://download.pytorch.org/whl/${CUDA_VERSION}"
fi

# Verify PyTorch installation
python -c "import torch; print(f'PyTorch {torch.__version__} installed successfully')"

echo ""
echo "Step 3: Installing MMCV..."
if [ "$CUDA_VERSION" = "cpu" ]; then
    pip install mmcv==${MMCV_VERSION} -f "https://download.openmmlab.com/mmcv/dist/cpu/torch${TORCH_VERSION}/index.html"
else
    pip install mmcv==${MMCV_VERSION} -f "https://download.openmmlab.com/mmcv/dist/${CUDA_VERSION}/torch${TORCH_VERSION}/index.html"
fi

echo ""
echo "Step 4: Installing MMDetection..."
pip install mmdet

echo ""
echo "Step 5: Installing GenEval dependencies..."
cd "$PROJECT_ROOT"
pip install -r requirements.txt

echo ""
echo "Step 6: Downloading object detection model..."
MODEL_DIR="${PROJECT_ROOT}/models"
mkdir -p "$MODEL_DIR"
"${PROJECT_ROOT}/evaluation/download_models.sh" "$MODEL_DIR/"

echo ""
echo "Step 7: Setting up MMDetection configs..."
"${SCRIPT_DIR}/setup_mmdet_configs.sh"

echo ""
echo "=============================================="
echo "Installation complete!"
echo "=============================================="
echo ""
echo "Verify installation:"
echo "  python -c \"import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')\""
echo "  python -c \"import mmdet; print(f'MMDetection: {mmdet.__version__}')\""
echo ""
echo "Quick test:"
echo "  python evaluation/evaluate_images.py --help"
echo ""
