#!/usr/bin/env python3
"""
Wrapper script for evaluate_images.py that automatically finds config files.
This is useful when mmdet is installed from PyPI without config files.

Usage:
    python scripts/evaluate.py <IMAGE_FOLDER> --outfile results.jsonl --model-path ./models

This wrapper will:
1. Try to find the config in mmdetection installation
2. Fall back to local configs/ directory
3. Download configs if not found (with user confirmation)
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path


def find_config_path() -> str:
    """Find Mask2Former config file."""
    
    # Option 1: Check mmdetection installation
    try:
        import mmdet
        mmdet_config = Path(mmdet.__file__).parent.parent / "configs" / "mask2former" / "mask2former_swin-s-p4-w7-224_lsj_8x2_50e_coco.py"
        if mmdet_config.exists():
            return str(mmdet_config)
    except ImportError:
        pass
    
    # Option 2: Check local configs directory
    project_root = Path(__file__).parent.parent
    local_config = project_root / "configs" / "mask2former" / "mask2former_swin-s-p4-w7-224_lsj_8x2_50e_coco.py"
    if local_config.exists():
        return str(local_config)
    
    # Option 3: Check if mmdetection repo is cloned nearby
    mmdet_repo = project_root / "mmdetection" / "configs" / "mask2former" / "mask2former_swin-s-p4-w7-224_lsj_8x2_50e_coco.py"
    if mmdet_repo.exists():
        return str(mmdet_repo)
    
    # Also try the new naming convention
    for base in [project_root / "configs", project_root / "mmdetection" / "configs"]:
        new_name = base / "mask2former" / "mask2former_swin-s-p4-w7-224_8xb2-lsj-50e_coco.py"
        if new_name.exists():
            return str(new_name)
    
    return None


def main():
    # Find config
    config_path = find_config_path()
    
    if config_path is None:
        print("Error: Could not find Mask2Former config file.", file=sys.stderr)
        print("", file=sys.stderr)
        print("Solutions:", file=sys.stderr)
        print("  1. Run: ./scripts/setup_mmdet_configs.sh", file=sys.stderr)
        print("  2. Or install mmdetection from source:", file=sys.stderr)
        print("     git clone https://github.com/open-mmlab/mmdetection.git", file=sys.stderr)
        print("     cd mmdetection && pip install -v -e .", file=sys.stderr)
        print("  3. Or specify config manually with --model-config", file=sys.stderr)
        sys.exit(1)
    
    print(f"Using config: {config_path}", file=sys.stderr)
    
    # Build command
    project_root = Path(__file__).parent.parent
    evaluate_script = project_root / "evaluation" / "evaluate_images.py"
    
    # Pass through all arguments, adding --model-config if not specified
    args = sys.argv[1:]
    
    if "--model-config" not in args:
        args = ["--model-config", config_path] + args
    
    cmd = [sys.executable, str(evaluate_script)] + args
    
    # Run evaluation
    result = subprocess.run(cmd)
    sys.exit(result.returncode)


if __name__ == "__main__":
    main()
