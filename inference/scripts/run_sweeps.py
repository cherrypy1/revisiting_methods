"""Run 1D HP sweeps over multiple methods sequentially on a single GPU.

Replaces ``run_sweeps.sh``. Sweep specs live in ``SWEEPS`` below — edit there
to add new sweeps rather than copy-paste the bash command.
"""

from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
SWEEP = PROJECT_ROOT / "scripts" / "sweep.py"
PROMPTS = PROJECT_ROOT / "prompts" / "evaluation_metadata.jsonl"
N_PROMPTS = 20
OUT_ROOT = PROJECT_ROOT / "outputs" / "sweep"


SWEEPS = [
    # (config_relpath, param, values, subdir)
    ("generation/configs/sd35/cfgpp.py",  "guidance_scale", [0.5, 0.7, 0.9, 1.2], "cfgpp_scale"),
    ("generation/configs/sd35/cfg0s.py",  "guidance_scale", [4.5, 5.5, 6.5],      "cfg0s_scale"),
    ("generation/configs/sd35/cfg0s.py",  "zero_steps",     [0, 1, 2],            "cfg0s_zsteps"),
    ("generation/configs/sd35/oseg.py",   "oseg_scale",     [0.5, 1.0, 1.5, 2.0], "oseg_oscale"),
    ("generation/configs/sd35/oseg.py",   "seg_scale",      [2.0, 3.0, 4.0],      "oseg_sscale"),
    ("generation/configs/sd35/tcfg.py",   "tcfg_rank",      [1, 2],               "tcfg_rank"),
    ("generation/configs/sd35/tcfg.py",   "guidance_scale", [4.0, 5.5, 7.0],      "tcfg_scale"),
    ("generation/configs/sd35/pag.py",    "pag_scale",      [1.0, 2.0, 3.0, 5.0], "pag_scale"),
]


def say(*msg):
    print(f"[{time.strftime('%F %T')}]", *msg, flush=True)


def main():
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    for cfg, param, values, sub in SWEEPS:
        say(f"=== {Path(cfg).stem} {param} ===")
        out = OUT_ROOT / sub
        cmd = [
            sys.executable, str(SWEEP), str(PROJECT_ROOT / cfg),
            "--param", param,
            "--values", *[str(v) for v in values],
            "--prompts", str(PROMPTS),
            "--n-prompts", str(N_PROMPTS),
            "--out-dir", str(out),
        ]
        rc = subprocess.call(cmd)
        if rc != 0:
            say(f"sweep failed for {sub} (exit {rc}) — stopping")
            sys.exit(rc)
    say("=== ALL SWEEPS DONE ===")


if __name__ == "__main__":
    main()
