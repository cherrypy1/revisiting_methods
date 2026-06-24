"""Re-run all benches for PAG only, with current pag.py / pag.yaml config.

Replaces ``run_pag_only.sh``. Runs on the compute node (assumes module purge
+ gnu14/14.1 loaded by caller). Uses ``scripts/bench.py`` — skip-existing is
expected to short-circuit gen on reruns.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
BENCH = PROJECT_ROOT / "scripts" / "bench.py"
TAG = "strat27042026"
METHOD = "pag"


def run(*args):
    cmd = [sys.executable, str(BENCH), METHOD, *map(str, args), "--run-tag", TAG]
    print(f"$ {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, check=True)


def main():
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    print("##### geneval #####", flush=True)
    run("geneval", "--limit", 90)
    print("##### oneig #####", flush=True)
    run("oneig", "--limit", 15, "--grid", "1x1")
    print("##### dpg #####", flush=True)
    run("dpg", "--limit", 90, "--pic-num", 1)
    print("ALL DONE", flush=True)


if __name__ == "__main__":
    main()
