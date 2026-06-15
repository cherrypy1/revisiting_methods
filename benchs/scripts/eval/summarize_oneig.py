"""Aggregate OneIG eval CSVs into one readable table.

OneIG score scripts append timestamped CSVs into ``results/`` (we copy them
into ``outputs/oneig/<tag>/eval/<method>/``). For each method we pick the
**latest** CSV per metric family and stitch the model-row into one summary.

Metric families used:
    alignment_score_EN_*.csv  -> 'alignment'  (object class)
    text_score_EN_*.csv       -> 'ED','CR','WAC','text score'
    reasoning_score_EN_*.csv  -> 'reasoning'

Old CSVs (e.g. dated before run start) and the per-prompt CSVs
(``*_prompt_score_*``) are ignored.

Usage:
    scripts/eval/summarize_oneig.py outputs/oneig/<tag>/eval
"""
from __future__ import annotations

import argparse
import csv
import re
import sys
from pathlib import Path

PATTERNS = {
    "alignment": re.compile(r"^alignment_score_EN_.*\.csv$"),
    "text":      re.compile(r"^text_score_EN_.*\.csv$"),
    "reasoning": re.compile(r"^reasoning_score_EN_.*\.csv$"),
}


def latest(method_dir: Path, regex: re.Pattern) -> Path | None:
    cands = [p for p in method_dir.iterdir() if regex.match(p.name)]
    return max(cands, key=lambda p: p.stat().st_mtime) if cands else None


def row_for(method: str, csv_path: Path) -> dict:
    with csv_path.open() as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        if (r.get("") or "").strip() == method:
            return {k: v for k, v in r.items() if k}
    return {}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("eval_root", help="outputs/oneig/<tag>/eval")
    args = ap.parse_args()
    root = Path(args.eval_root)
    if not root.is_dir():
        sys.exit(f"not a dir: {root}")

    methods = sorted(p.name for p in root.iterdir() if p.is_dir())
    if not methods:
        sys.exit(f"no method subdirs in {root}")

    rows = []
    for m in methods:
        mdir = root / m
        rec = {"method": m}
        for fam, rx in PATTERNS.items():
            f = latest(mdir, rx)
            if f is None:
                continue
            for k, v in row_for(m, f).items():
                rec[f"{fam}.{k}"] = v
        rows.append(rec)

    cols = ["method"]
    seen = set(cols)
    for r in rows:
        for k in r:
            if k not in seen:
                cols.append(k); seen.add(k)

    widths = {c: max(len(c), max((len(str(r.get(c, ""))) for r in rows), default=0)) for c in cols}
    print(" | ".join(c.ljust(widths[c]) for c in cols))
    print("-+-".join("-" * widths[c] for c in cols))
    for r in rows:
        print(" | ".join(str(r.get(c, "")).ljust(widths[c]) for c in cols))


if __name__ == "__main__":
    main()
