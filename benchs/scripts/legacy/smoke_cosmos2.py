"""Smoke-test all Cosmos2 methods on 2 geneval prompts each.

Validates that every config + pipeline path runs. Replaces ``smoke_cosmos2.sh``.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
METHODS = ["no_cfg", "cfg", "cfgpp", "cfg0s", "apg", "tcfg", "pag", "sag", "seg_sigma10", "oseg"]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run-tag", default="smoke_cosmos2_" + time.strftime("%d%m%Y_%H%M"))
    p.add_argument("--limit", type=int, default=2)
    p.add_argument("--methods", nargs="+", default=METHODS)
    args = p.parse_args()

    log_dir = PROJECT_ROOT / "outputs" / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    failures = []
    for m in args.methods:
        print(f"### smoke {m} ###", flush=True)
        log = log_dir / f"cosmos2_smoke_{m}.log"
        cmd = [
            sys.executable, str(PROJECT_ROOT / "scripts" / "bench.py"),
            m, "geneval", "--model", "cosmos2",
            "--limit", str(args.limit), "--run-tag", args.run_tag,
            "--skip-eval",
        ]
        with open(log, "w") as lf:
            rc = subprocess.call(cmd, stdout=lf, stderr=subprocess.STDOUT)
        if rc != 0:
            print(f"FAILED {m}  (log: {log})")
            failures.append(m)

    print("DONE")
    if failures:
        print(f"  failures: {failures}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
