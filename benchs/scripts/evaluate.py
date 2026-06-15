"""Mode-aware benchmark orchestrator.

This is the public Python entrypoint behind:

    bash scripts/smoke_test.sh <model> <bench...> <method...> [-- extra args]
    bash scripts/evaluation.sh <model> <bench...> <method...> [-- extra args]
    bash scripts/full_test.sh <model> <bench...> <method...> [-- extra args]

Each (method, bench) pair is delegated to scripts/bench.py. Prompt selection is
done by --prompt-set and therefore depends on files prepared by
scripts/prepare_benchmarks.sh.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
BENCH_SCRIPT = PROJECT_ROOT / "scripts" / "bench.py"
BENCHES = ("geneval", "oneig", "dpg")
MODES = ("smoke_test", "evaluation", "full_test")


def expand_benches(values: list[str]) -> list[str]:
    expanded: list[str] = []
    for value in values:
        if value == "all":
            candidates = BENCHES
        elif value in BENCHES:
            candidates = (value,)
        else:
            raise SystemExit(f"Unknown bench: {value}")
        for bench in candidates:
            if bench not in expanded:
                expanded.append(bench)
    return expanded


def run_one(
    *,
    mode: str,
    model: str,
    method: str,
    bench: str,
    run_tag: str,
    passthrough: list[str],
    dry_run: bool,
) -> int:
    cmd = [
        sys.executable,
        str(BENCH_SCRIPT),
        method,
        bench,
        "--model",
        model,
        "--prompt-set",
        mode,
        "--out-root",
        str(PROJECT_ROOT / "outputs" / mode),
        "--run-tag",
        run_tag,
        "--n_samples",
        "4",
        "--grid",
        "2x2",
        "--pic-num",
        "4",
    ]
    cmd += passthrough

    ts = time.strftime("%F %T")
    print(f"[{ts}] === {mode} :: {model} :: {method} :: {bench} ===", flush=True)
    print("$ " + " ".join(str(part) for part in cmd), flush=True)
    if dry_run:
        return 0
    return subprocess.call(cmd)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", required=True, choices=MODES)
    parser.add_argument("--model", required=True, choices=["sd35", "flux2_klein_base", "cosmos2"])
    parser.add_argument("--benches", nargs="+", required=True)
    parser.add_argument("--methods", nargs="+", required=True)
    parser.add_argument("--run-tag", default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--keep-going", action="store_true")
    parser.add_argument(
        "passthrough",
        nargs=argparse.REMAINDER,
        help="Extra args passed to every bench.py call after `--`.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    benches = expand_benches(args.benches)
    run_tag = args.run_tag or f"{args.mode}_{datetime.now().strftime('%d%m%Y')}"

    passthrough = args.passthrough
    if passthrough and passthrough[0] == "--":
        passthrough = passthrough[1:]

    failures: list[tuple[str, str, int]] = []
    for method in args.methods:
        for bench in benches:
            rc = run_one(
                mode=args.mode,
                model=args.model,
                method=method,
                bench=bench,
                run_tag=run_tag,
                passthrough=passthrough,
                dry_run=args.dry_run,
            )
            if rc != 0:
                failures.append((method, bench, rc))
                if not args.keep_going:
                    print(f"[FAIL] {method}:{bench} exit {rc}", file=sys.stderr)
                    sys.exit(rc)

    if failures:
        for method, bench, rc in failures:
            print(f"[FAIL] {method}:{bench} exit {rc}", file=sys.stderr)
        sys.exit(1)
    print("=== ALL DONE ===")


if __name__ == "__main__":
    main()
