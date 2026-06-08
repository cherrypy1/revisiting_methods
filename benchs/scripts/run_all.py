"""Run every method config on one or more benchmarks.

Usage:
    scripts/run_all.py <bench> [--methods m1 m2 ...] [--skip-eval] [--limit N]

Methods default to every ``.yaml`` in ``configs/sd35/`` excluding
helpers prefixed with ``_``.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR = PROJECT_ROOT / "configs" / "sd35"


def all_methods():
    return sorted(p.stem for p in CONFIG_DIR.glob("*.yaml") if not p.stem.startswith("_"))


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("bench", choices=["geneval", "oneig", "dpg"])
    p.add_argument("--methods", nargs="+", default=None)
    p.add_argument("--skip-eval", action="store_true")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--run-tag", default=None)
    p.add_argument("--grid", default="2x2", choices=["1x1", "2x2"])
    p.add_argument("--pic-num", type=int, default=4)
    p.add_argument("--resolution", type=int, default=1024)
    p.add_argument("--continue-on-error", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    methods = args.methods or all_methods()

    for method in methods:
        print(f"=== {method} on {args.bench} ===", flush=True)
        cmd = [sys.executable, str(PROJECT_ROOT / "scripts" / "bench.py"),
               method, args.bench, "--grid", args.grid,
               "--pic-num", str(args.pic_num), "--resolution", str(args.resolution)]
        if args.run_tag:
            cmd += ["--run-tag", args.run_tag]
        if args.limit:
            cmd += ["--limit", str(args.limit)]
        if args.skip_eval:
            cmd += ["--skip-eval"]
        try:
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError as e:
            if args.continue_on_error:
                print(f"[warn] {method} failed: {e}", file=sys.stderr)
                continue
            raise


if __name__ == "__main__":
    main()
