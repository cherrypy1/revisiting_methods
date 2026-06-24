"""Aggregate OneIG eval CSVs into the final per-method comparison table.

OneIG's score scripts append timestamped CSVs into ``results/`` (we copy them
into ``outputs[/<model>]/oneig/<tag>/eval/<method>/``). For each method we pick
the **latest** CSV per metric family and read that method's row.

Families and the column we treat as the *final* score:
    alignment_score_EN_*.csv  -> 'alignment'    (General_Object alignment)
    text_score_EN_*.csv       -> 'text score'   (composite; ED/CR/WAC are raw)
    reasoning_score_EN_*.csv  -> 'reasoning'    (Knowledge_Reasoning)

All final scores are on a 0-1 scale. ``--raw`` additionally dumps the raw text
sub-metrics (ED = edit distance, CR = completion rate, WAC = word accuracy) and
any other family columns present.

Old CSVs and the per-prompt CSVs (``*_prompt_score_*``) are ignored.

Usage:
    scripts/summarize_oneig.py <tag> [--model sd35|cosmos2] [--raw] [--csv FILE]
    scripts/summarize_oneig.py --eval-root outputs/cosmos2/oneig/<tag>/eval
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# family -> (filename regex, csv column holding that family's FINAL score)
FAMILIES = {
    "object": (re.compile(r"^alignment_score_EN_.*\.csv$"), "alignment"),
    "text":   (re.compile(r"^text_score_EN_.*\.csv$"),      "text score"),
    "reason": (re.compile(r"^reasoning_score_EN_.*\.csv$"), "reasoning"),
}
# raw text sub-metrics (only shown with --raw)
TEXT_RAW = ["ED", "CR", "WAC"]


def latest(method_dir: Path, regex: re.Pattern):
    cands = [p for p in method_dir.iterdir() if regex.match(p.name)]
    return max(cands, key=lambda p: p.stat().st_mtime) if cands else None


def row_for(method: str, csv_path: Path) -> dict:
    """Row whose unnamed first column == method, as {column: value}."""
    with csv_path.open() as f:
        for r in csv.DictReader(f):
            if (r.get("") or "").strip() == method:
                return {k: v for k, v in r.items() if k}
    return {}


def fnum(s):
    try:
        return f"{float(s):.3f}"
    except (TypeError, ValueError):
        return "-"


def collect(eval_root: Path, raw: bool):
    methods = sorted(p.name for p in eval_root.iterdir() if p.is_dir())
    if not methods:
        sys.exit(f"no method subdirs in {eval_root}")

    cols = ["object", "text", "reason"] + (TEXT_RAW if raw else [])
    rows = []
    for m in methods:
        mdir = eval_root / m
        rec = {"method": m}
        text_row = {}
        for fam, (rx, score_col) in FAMILIES.items():
            f = latest(mdir, rx)
            if f is None:
                continue
            data = row_for(m, f)
            rec[fam] = fnum(data.get(score_col))
            if fam == "text":
                text_row = data
        if raw:
            for k in TEXT_RAW:
                v = text_row.get(k)
                # ED is a raw edit distance (not 0-1); round for readability.
                try:
                    rec[k] = f"{float(v):.1f}" if k == "ED" else f"{float(v):.3f}"
                except (TypeError, ValueError):
                    rec[k] = "-"
        rows.append(rec)
    return cols, rows


def print_table(cols, rows):
    header = ["method"] + cols
    body = [[r["method"]] + [str(r.get(c, "-")) for c in cols] for r in rows]
    table = [header] + body
    widths = [max(len(t[i]) for t in table) for i in range(len(header))]

    def line(r):
        return "  ".join(c.rjust(widths[i]) if i else c.ljust(widths[i])
                         for i, c in enumerate(r))

    print(line(header))
    print("-" * (sum(widths) + 2 * (len(widths) - 1)))
    for r in body:
        print(line(r))
    print("\n(all final scores 0-1; object=alignment, text=composite, "
          "reason=reasoning)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("tag", nargs="?", help="run-tag (e.g. cosmos2_24052026)")
    ap.add_argument("--model", default="sd35", choices=["sd35", "cosmos2"])
    ap.add_argument("--eval-root", default=None,
                    help="explicit eval dir; overrides tag/model")
    ap.add_argument("--out-root", default=None,
                    help="base outputs dir (default: <repo>/outputs)")
    ap.add_argument("--raw", action="store_true",
                    help="also show raw text sub-metrics (ED/CR/WAC)")
    ap.add_argument("--csv", default=None, help="also write the table as CSV")
    args = ap.parse_args()

    if args.eval_root:
        eval_root = Path(args.eval_root)
    elif args.tag:
        base = Path(args.out_root) if args.out_root else PROJECT_ROOT / "outputs"
        root = base / args.model if args.model != "sd35" else base
        eval_root = root / "oneig" / args.tag / "eval"
    else:
        ap.error("give a <tag> or --eval-root")
    if not eval_root.is_dir():
        sys.exit(f"not a dir: {eval_root}")

    cols, rows = collect(eval_root, args.raw)
    print_table(cols, rows)

    if args.csv:
        with open(args.csv, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["method"] + cols)
            for r in rows:
                w.writerow([r["method"]] + [r.get(c, "") for c in cols])
        print(f"wrote {args.csv}")


if __name__ == "__main__":
    main()
