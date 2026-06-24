"""Generic batched bench runner: each (method, bench) pair as a bench.py call.

Usage:
    python scripts/run_methods.py --methods cfg pag sag \
        --benches geneval oneig dpg \
        --run-tag strat27042026 \
        --limits geneval=90 oneig=15 dpg=90 \
        --extra oneig:--grid=1x1 dpg:--pic-num=1

Replaces ad-hoc bash loops (run_strat.sh, run_round2.sh, run_final.sh,
run_all_methods.sh, run_pag_only.sh). Each call shells out to
``scripts/bench.py``; failures stop the loop unless --keep-going.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
BENCH = PROJECT_ROOT / "scripts" / "bench.py"


def parse_kv(items, sep="="):
    out = {}
    for item in items or []:
        if sep not in item:
            sys.exit(f"Expected KEY{sep}VAL, got {item!r}")
        k, v = item.split(sep, 1)
        out[k] = v
    return out


def parse_extra(items):
    """Parse ``bench:flag=val`` overrides like ``oneig:--grid=1x1``."""
    out: dict[str, list[str]] = {}
    for item in items or []:
        if ":" not in item:
            sys.exit(f"Expected BENCH:FLAG, got {item!r}")
        bench, flag = item.split(":", 1)
        out.setdefault(bench, []).extend(flag.split())
    return out


def run_one(method, bench, run_tag, limits, extra, dry_run, model="sd35"):
    cmd = [sys.executable, str(BENCH), method, bench, "--run-tag", run_tag,
           "--model", model]
    if bench in limits:
        cmd += ["--limit", limits[bench]]
    cmd += extra.get(bench, [])
    ts = time.strftime("%F %T")
    print(f"[{ts}] === {method} :: {bench} ===  $ {' '.join(cmd)}", flush=True)
    if dry_run:
        return 0
    return subprocess.call(cmd)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--methods", nargs="+", required=True)
    p.add_argument("--benches", nargs="+", default=["geneval", "oneig", "dpg"])
    p.add_argument("--model", default="sd35", choices=["sd35", "cosmos2"])
    p.add_argument("--run-tag", required=True)
    p.add_argument("--limits", nargs="*", default=[], help="bench=N pairs")
    p.add_argument("--extra", nargs="*", default=[],
                   help="bench:flag pairs, e.g. oneig:--grid=1x1")
    p.add_argument("--keep-going", action="store_true",
                   help="Continue on per-call failure")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    limits = parse_kv(args.limits)
    extra = parse_extra(args.extra)

    failures: list[tuple[str, str, int]] = []
    for method in args.methods:
        for bench in args.benches:
            rc = run_one(method, bench, args.run_tag, limits, extra, args.dry_run,
                         model=args.model)
            if rc != 0:
                failures.append((method, bench, rc))
                if not args.keep_going:
                    print(f"[FAIL] {method}:{bench} exit {rc} — stopping", file=sys.stderr)
                    sys.exit(rc)

    print("=== ALL DONE ===")
    if failures:
        for m, b, rc in failures:
            print(f"  failed: {m}:{b} (exit {rc})", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
