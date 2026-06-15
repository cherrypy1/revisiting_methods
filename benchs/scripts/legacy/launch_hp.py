"""Wrapper: run Cosmos HP sweep on a single GPU with given methods.

Replaces ``launch_hp.sh``. Sets CUDA_VISIBLE_DEVICES, runs
``scripts/legacy/cosmos_hp_run.py`` with stdout/stderr redirected to a per-GPU log.

Usage:
    python scripts/legacy/launch_hp.py <gpu_idx> <tag> [method1 method2 ...]
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_VENV_PYTHON = Path.home() / ".venv" / "bin" / "python"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("gpu", type=int)
    p.add_argument("tag")
    p.add_argument("methods", nargs="+")
    p.add_argument("--per-tag", type=int, default=2)
    p.add_argument("--steps", type=int, default=20)
    p.add_argument("--python", default=str(DEFAULT_VENV_PYTHON))
    args = p.parse_args()

    log_dir = PROJECT_ROOT / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log = log_dir / f"hp_{args.tag}_gpu{args.gpu}.log"

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    cmd = [
        args.python,
        str(PROJECT_ROOT / "scripts" / "legacy" / "cosmos_hp_run.py"),
        "--methods", *args.methods,
        "--per-tag", str(args.per_tag),
        "--steps", str(args.steps),
    ]

    print(f"[launch] GPU={args.gpu} TAG={args.tag} methods={args.methods} -> {log}")
    with open(log, "w") as lf:
        sys.exit(subprocess.call(cmd, env=env, stdout=lf, stderr=subprocess.STDOUT,
                                 cwd=str(PROJECT_ROOT)))


if __name__ == "__main__":
    main()
