"""Run ELLA/dpg_bench evaluation on a generated DPG-Bench image set.

Upstream ``dpg_bench/dist_eval.sh`` defaults ``--multi_gpu --num_processes 8
--pic-num 4`` which breaks on a single V100 and misaligns with our
``pic-num=1`` generation. We invoke ``compute_dpg_bench.py`` directly under
``accelerate launch --num_processes 1`` with the correct ``--pic-num``.

Usage:
    scripts/eval/eval_dpg.py --dpg-root $DPG_ROOT \\
        --image-dir outputs/dpg/<method>_<tag>/images \\
        --pic-num 1 --resolution 1024 \\
        --out-dir outputs/dpg/<method>_<tag>/eval
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dpg-root", required=True, help="Path to ELLA repo")
    p.add_argument("--dpg-python", default=None,
                   help="Python executable with DPG evaluator deps")
    p.add_argument("--image-dir", required=True,
                   help="Folder holding {id}.png files (listdir'd directly)")
    p.add_argument("--pic-num", type=int, default=1)
    p.add_argument("--resolution", type=int, default=1024)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--csv", default=None, help="Optional DPG subset csv matching generated prompt ids")
    p.add_argument("--port", default="29500")
    return p.parse_args()


def run(cmd, **kw):
    print(f"$ {' '.join(str(c) for c in cmd)}", flush=True)
    subprocess.run([str(c) for c in cmd], check=True, **kw)


def resolve_bench_python(arg_value: str | None) -> str:
    candidates = [
        arg_value,
        os.environ.get("DPG_PYTHON"),
        os.environ.get("BENCH_PYTHON"),
        str(PROJECT_ROOT / ".venv_bench" / "bin" / "python"),
    ]
    for candidate in candidates:
        if candidate and Path(candidate).is_file():
            return candidate
    raise SystemExit(
        "DPG evaluator python not found. Run "
        "`bash scripts/setup/install_dpg_eval.sh` or set BENCH_PYTHON."
    )


def main():
    args = parse_args()
    dpg_root = Path(args.dpg_root).resolve()
    image_dir = Path(args.image_dir).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    compute = dpg_root / "dpg_bench" / "compute_dpg_bench.py"
    if not compute.is_file():
        sys.exit(f"compute_dpg_bench.py missing: {compute}")

    bench_python = resolve_bench_python(args.dpg_python)
    accelerate_bin = Path(bench_python).parent / "accelerate"
    if accelerate_bin.is_file():
        accelerate_cmd = [str(accelerate_bin), "launch"]
    else:
        accelerate_cmd = [bench_python, "-m", "accelerate.commands.launch"]

    res_path = out_dir / "dpg_score.txt"

    env = os.environ.copy()
    cmd = [
        *accelerate_cmd,
        "--num_machines", "1",
        "--num_processes", "1",
        "--mixed_precision", "fp16",
        "--main_process_port", args.port,
        str(compute),
        "--image-root-path", str(image_dir),
        "--resolution", str(args.resolution),
        "--pic-num", str(args.pic_num),
        "--vqa-model", "mplug",
        "--res-path", str(res_path),
    ]
    if args.csv:
        cmd += ["--csv", str(Path(args.csv).resolve())]
    run(cmd, cwd=dpg_root, env=env)

    print(f"[dpg eval] scores -> {out_dir}")


if __name__ == "__main__":
    main()
