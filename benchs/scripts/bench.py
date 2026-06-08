"""Unified benchmark driver: generate images for one method on one bench,
then (optionally) run that bench's evaluator into the same folder.

Usage:
    scripts/bench.py <method> <geneval|oneig|dpg> [--skip-eval] [--limit N]
                     [--run-tag TAG] [--grid 1x1|2x2]  # oneig only
                     [--pic-num N] [--resolution N]     # dpg only
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
# GenEval benchmark assets (detector .pth, mmdet configs, prompts, scorer) live
# in a sibling repo — like OneIG-Benchmark / ELLA. Override with $GENEVAL_ROOT.
GENEVAL_ROOT = Path(os.environ.get("GENEVAL_ROOT", str(Path.home() / "geneval-bench")))


def env_path(name, default):
    return Path(os.environ.get(name, default))


def run(cmd, **kw):
    print(f"$ {' '.join(str(c) for c in cmd)}", flush=True)
    subprocess.run([str(c) for c in cmd], check=True, **kw)


def config_path(method, model="sd35"):
    # Canonical config: configs/<model>/<method>.yaml.
    cfg = PROJECT_ROOT / "configs" / model / f"{method}.yaml"
    if cfg.exists():
        return cfg
    sys.exit(f"Config not found: {cfg}")


def run_geneval(args, cfg, out_root):
    prompts = env_path("GENEVAL_PROMPTS", GENEVAL_ROOT / "prompts" / "evaluation_metadata.jsonl")
    out = out_root / "geneval" / f"{args.method}_{args.run_tag}"
    out.mkdir(parents=True, exist_ok=True)

    gen_cmd = [sys.executable, PROJECT_ROOT / "generation" / "diffusers_generate.py",
               prompts, "--config", cfg, "--outdir", out]
    if args.limit:
        gen_cmd += ["--limit", args.limit]
    run(gen_cmd)

    if args.skip_eval:
        return
    models = env_path("GENEVAL_MODELS", GENEVAL_ROOT / "models")
    mm_config = os.environ.get(
        "GENEVAL_MM_CONFIG",
        str(GENEVAL_ROOT / "mmdetection/configs/mask2former/"
                          "mask2former_swin-s-p4-w7-224_8xb2-lsj-50e_coco.py"),
    )
    run([sys.executable, PROJECT_ROOT / "scripts" / "eval_geneval.py",
         "--image-dir", out, "--models-dir", models, "--mm-config", mm_config])


def run_oneig(args, cfg, out_root):
    csv = env_path("ONEIG_CSV", Path.home() / "OneIG-Benchmark" / "OneIG-Bench.csv")
    out = out_root / "oneig" / args.run_tag
    out.mkdir(parents=True, exist_ok=True)

    gen_cmd = [sys.executable, PROJECT_ROOT / "scripts" / "generate_oneig.py", cfg,
               "--csv", csv, "--out-dir", out, "--model-name", args.method,
               "--grid", args.grid]
    if args.limit:
        gen_cmd += ["--limit", args.limit]
    run(gen_cmd)

    if args.skip_eval:
        return
    oneig_root = env_path("ONEIG_ROOT", Path.home() / "OneIG-Benchmark")
    eval_cmd = [sys.executable, PROJECT_ROOT / "scripts" / "eval_oneig.py",
                "--oneig-root", oneig_root, "--image-dir", out,
                "--model-name", args.method, "--grid", args.grid,
                "--out-dir", out / "eval" / args.method]
    run(eval_cmd)


def run_dpg(args, cfg, out_root):
    prompts_dir = env_path("DPG_PROMPTS", Path.home() / "ELLA" / "dpg_bench" / "prompts")
    out = out_root / "dpg" / f"{args.method}_{args.run_tag}"
    images = out / "images"
    images.mkdir(parents=True, exist_ok=True)

    gen_cmd = [sys.executable, PROJECT_ROOT / "scripts" / "generate_dpg.py", cfg,
               "--prompts-dir", prompts_dir, "--out-dir", images,
               "--pic-num", args.pic_num, "--resolution", args.resolution]
    if args.limit:
        gen_cmd += ["--limit", args.limit]
    run(gen_cmd)

    if args.skip_eval:
        return
    dpg_root = env_path("DPG_ROOT", Path.home() / "ELLA")
    eval_cmd = [sys.executable, PROJECT_ROOT / "scripts" / "eval_dpg.py",
                "--dpg-root", dpg_root, "--image-dir", images,
                "--pic-num", args.pic_num,
                "--resolution", args.resolution, "--out-dir", out / "eval"]
    run(eval_cmd)


def parse_args():
    from datetime import datetime
    p = argparse.ArgumentParser()
    p.add_argument("method")
    p.add_argument("bench", choices=["geneval", "oneig", "dpg"])
    p.add_argument("--model", default="sd35", choices=["sd35", "cosmos2"])
    p.add_argument("--run-tag", default=datetime.now().strftime("%d%m%Y"))
    p.add_argument("--out-root", default=None)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--skip-eval", action="store_true")
    p.add_argument("--grid", default="2x2", choices=["1x1", "2x2"], help="oneig only")
    p.add_argument("--pic-num", type=int, default=4, help="dpg only")
    p.add_argument("--resolution", type=int, default=1024, help="dpg only")
    return p.parse_args()


def main():
    args = parse_args()
    cfg = config_path(args.method, args.model)
    base_out = Path(args.out_root) if args.out_root else PROJECT_ROOT / "outputs"
    out_root = base_out / args.model if args.model != "sd35" else base_out
    out_root.mkdir(parents=True, exist_ok=True)

    dispatch = {"geneval": run_geneval, "oneig": run_oneig, "dpg": run_dpg}
    dispatch[args.bench](args, cfg, out_root)


if __name__ == "__main__":
    main()
