#!/usr/bin/env python3
"""
Verification script to check GenEval installation
Run this after installation to verify everything works correctly.
"""

import sys
import importlib
from typing import List, Tuple


def check_import(module_name: str, min_version: str = None) -> Tuple[bool, str]:
    """Check if a module can be imported and optionally verify version."""
    try:
        module = importlib.import_module(module_name)
        version = getattr(module, '__version__', 'unknown')
        
        if min_version and version != 'unknown':
            from packaging import version as pkg_version
            if pkg_version.parse(version) < pkg_version.parse(min_version):
                return False, f"{module_name} {version} (requires >= {min_version})"
        
        return True, f"{module_name} {version}"
    except ImportError as e:
        return False, f"{module_name} - MISSING ({e})"


def check_cuda() -> Tuple[bool, str]:
    """Check CUDA availability."""
    try:
        import torch
        if torch.cuda.is_available():
            return True, f"CUDA available (device: {torch.cuda.get_device_name(0)})"
        else:
            return False, "CUDA not available (CPU mode only - evaluation may be slow)"
    except Exception as e:
        return False, f"CUDA check failed: {e}"


def check_mmdet_config() -> Tuple[bool, str]:
    """Check if MMDetection config file exists."""
    import os
    
    try:
        import mmdet
        config_path = os.path.join(
            os.path.dirname(mmdet.__file__),
            "../configs/mask2former/mask2former_swin-s-p4-w7-224_lsj_8x2_50e_coco.py"
        )
        
        # Also check local configs directory
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(script_dir)
        local_config = os.path.join(
            project_root, 
            "configs/mask2former/mask2former_swin-s-p4-w7-224_lsj_8x2_50e_coco.py"
        )
        
        if os.path.exists(config_path):
            return True, f"Config found at mmdet installation"
        elif os.path.exists(local_config):
            return True, f"Config found at local configs/"
        else:
            return False, "Mask2Former config not found (run scripts/setup_mmdet_configs.sh or install mmdet from source)"
    except Exception as e:
        return False, f"Config check failed: {e}"


def check_model_weights() -> Tuple[bool, str]:
    """Check if model weights are downloaded."""
    import os
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    model_path = os.path.join(project_root, "models/mask2former_swin-s-p4-w7-224_lsj_8x2_50e_coco.pth")
    
    if os.path.exists(model_path):
        size_mb = os.path.getsize(model_path) / (1024 * 1024)
        return True, f"Model weights found ({size_mb:.1f} MB)"
    else:
        return False, "Model weights not found (run: ./evaluation/download_models.sh ./models/)"


def main():
    print("=" * 60)
    print("GenEval Installation Verification")
    print("=" * 60)
    print()

    checks = [
        ("Core Dependencies", [
            ("torch", "2.0.0"),
            ("torchvision", "0.15.0"),
            ("numpy", "1.23.0"),
            ("pandas", "1.5.0"),
            ("PIL", None),  # Pillow
        ]),
        ("MMDetection Stack", [
            ("mmcv", "2.0.0"),
            ("mmdet", "3.0.0"),
            ("mmengine", None),
        ]),
        ("Generation Dependencies", [
            ("diffusers", "0.24.0"),
            ("transformers", "4.36.0"),
            ("accelerate", None),
        ]),
        ("Evaluation Dependencies", [
            ("open_clip", None),
            ("clip_benchmark", None),
        ]),
        ("Utilities", [
            ("einops", None),
            ("tqdm", None),
            ("yaml", None),
        ]),
    ]

    all_passed = True
    
    for category, modules in checks:
        print(f"\n{category}:")
        print("-" * 40)
        for module_name, min_version in modules:
            passed, message = check_import(module_name, min_version)
            status = "✓" if passed else "✗"
            print(f"  {status} {message}")
            if not passed and module_name in ['torch', 'mmdet', 'mmcv', 'open_clip']:
                all_passed = False

    print(f"\nEnvironment Checks:")
    print("-" * 40)
    
    # CUDA check
    passed, message = check_cuda()
    status = "✓" if passed else "⚠"
    print(f"  {status} {message}")
    
    # Config check
    passed, message = check_mmdet_config()
    status = "✓" if passed else "✗"
    print(f"  {status} {message}")
    if not passed:
        all_passed = False
    
    # Model weights check
    passed, message = check_model_weights()
    status = "✓" if passed else "✗"
    print(f"  {status} {message}")
    if not passed:
        all_passed = False

    print()
    print("=" * 60)
    if all_passed:
        print("✓ All critical checks passed! GenEval is ready to use.")
    else:
        print("✗ Some checks failed. Please review the issues above.")
        print("  See INSTALL.md for troubleshooting guidance.")
    print("=" * 60)
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
