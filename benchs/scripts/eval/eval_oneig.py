"""Run OneIG-Benchmark evaluation on a generated image set.

OneIG's upstream ``run_overall.sh`` hardcodes model/grid/class_items and
pins ``transformers==4.50.0``. We call the three relevant score modules
directly instead: alignment (object), text, reasoning. ``cwd=$ONEIG_ROOT``
so ``scripts.*`` resolve and ``qwen_vl_utils.py`` (sibling of scripts/) is
importable.

OneIG score scripts write CSVs into ``$ONEIG_ROOT/results/`` (hardcoded).
We copy any produced file that contains ``<model_name>`` into ``out_dir``.

Usage:
    scripts/eval/eval_oneig.py --oneig-root $ONEIG_ROOT \\
        --image-dir outputs/oneig/<run_tag> --model-name cfg \\
        --grid 1x1 --out-dir outputs/oneig/<run_tag>/eval/cfg
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path


GRID_MAP = {"1x1": "1,1", "2x2": "2,2"}
PROJECT_ROOT = Path(__file__).resolve().parents[2]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--oneig-root", required=True)
    p.add_argument("--oneig-python", default=None,
                   help="Python executable with OneIG evaluator deps")
    p.add_argument("--image-dir", required=True,
                   help="Gen root: {image_dir}/{object,text,reasoning}/{model}/*.webp")
    p.add_argument("--model-name", required=True)
    p.add_argument("--grid", choices=list(GRID_MAP.keys()), default="1x1")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--mode", default="EN")
    return p.parse_args()


def run(cmd, **kw):
    print(f"$ {' '.join(str(c) for c in cmd)}", flush=True)
    subprocess.run([str(c) for c in cmd], check=True, **kw)


def resolve_oneig_python(arg_value: str | None) -> str:
    candidates = [
        arg_value,
        os.environ.get("ONEIG_PYTHON"),
        os.environ.get("BENCH_PYTHON"),
        str(PROJECT_ROOT / ".venv_bench" / "bin" / "python"),
    ]
    for candidate in candidates:
        if candidate and Path(candidate).is_file():
            return candidate
    raise SystemExit(
        "OneIG evaluator python not found. Run "
        "`bash scripts/setup/install_oneig_eval.sh` or set BENCH_PYTHON."
    )


def main():
    args = parse_args()
    oneig = Path(args.oneig_root).resolve()
    image_dir = Path(args.image_dir).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    if not (oneig / "scripts").is_dir():
        sys.exit(f"OneIG scripts dir missing: {oneig / 'scripts'}")

    oneig_python = resolve_oneig_python(args.oneig_python)
    grid = GRID_MAP[args.grid]
    env = os.environ.copy()

    common = [
        "--mode", args.mode,
        "--model_names", args.model_name,
        "--image_grid", grid,
    ]

    # alignment: iterates class_items; we only produce `object`
    run([oneig_python, "-m", "scripts.alignment.alignment_score",
         "--image_dirname", str(image_dir),
         "--class_items", "object",
         *common], cwd=oneig, env=env)

    # text: reads image_dirname/<model>/<id>*
    run([oneig_python, "-m", "scripts.text.text_score",
         "--image_dirname", str(image_dir / "text"),
         *common], cwd=oneig, env=env)

    # reasoning: reads image_dirname/<model>/*
    run([oneig_python, "-m", "scripts.reasoning.reasoning_score",
         "--image_dirname", str(image_dir / "reasoning"),
         *common], cwd=oneig, env=env)

    results = oneig / "results"
    if results.is_dir():
        for f in results.iterdir():
            if f.is_file():
                shutil.copy2(f, out_dir / f.name)

    print(f"[oneig eval] scores -> {out_dir}")


if __name__ == "__main__":
    main()
