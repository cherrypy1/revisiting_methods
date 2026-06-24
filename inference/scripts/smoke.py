"""Fast smoke check: generate N images per method per bench, no evaluation.

Usage:
    scripts/smoke.py                       # all benches, all methods
    scripts/smoke.py --benches geneval --methods cfg pag
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR = PROJECT_ROOT / "generation" / "configs" / "sd35"

DEFAULT_METHODS = ["cfg", "seg_sigma10_cfg3", "pag", "sag", "oseg", "tcfg", "cfgpp", "cfg0s"]
DEFAULT_BENCHES = ["geneval", "oneig", "dpg"]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--methods", nargs="+", default=DEFAULT_METHODS)
    p.add_argument("--benches", nargs="+", default=DEFAULT_BENCHES,
                   choices=DEFAULT_BENCHES)
    p.add_argument("--limit", type=int, default=25,
                   help="GenEval/DPG prompts or OneIG prompts per category")
    p.add_argument("--oneig-per-cat", type=int, default=2,
                   help="OneIG 2x2 grids contain 4 samples; keep low to hit target image count")
    p.add_argument("--grid", default="2x2", choices=["1x1", "2x2"])
    p.add_argument("--run-tag", default=f"smoke_{datetime.now().strftime('%d%m%Y_%H%M')}")
    p.add_argument("--out-root", default=None)
    p.add_argument("--continue-on-error", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    out_root = args.out_root or str(PROJECT_ROOT / "outputs")

    for bench in args.benches:
        limit = args.oneig_per_cat if bench == "oneig" else args.limit
        for method in args.methods:
            print(f"=== smoke: {method} / {bench} ===", flush=True)
            cmd = [sys.executable, str(PROJECT_ROOT / "scripts" / "bench.py"),
                   method, bench, "--skip-eval",
                   "--run-tag", args.run_tag,
                   "--out-root", out_root,
                   "--limit", str(limit),
                   "--grid", args.grid]
            try:
                subprocess.run(cmd, check=True)
            except subprocess.CalledProcessError as e:
                if args.continue_on_error:
                    print(f"[warn] {method}/{bench} failed: {e}", file=sys.stderr)
                    continue
                raise

    print(f"Smoke outputs: {out_root}/{args.run_tag}")


if __name__ == "__main__":
    main()
