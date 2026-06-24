"""Aggregate DPG-Bench ``dpg_score.txt`` files into one per-method table.

ELLA's ``compute_dpg_bench.py`` writes, per method, an ``eval/dpg_score.txt``
with one line per image: ``<image_path>, <score>, <score>``. The headline
DPG-Bench number is the mean of those per-image scores, ×100.

For every ``outputs[/<model>]/dpg/<method>_<tag>/eval/dpg_score.txt`` it prints
the final score and the image count (N should be the prompt count, e.g. 90).

Usage:
    scripts/summarize_dpg.py <tag> [--model sd35|cosmos2]
                             [--out-root DIR] [--csv FILE]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
METHOD_ORDER = ["no_cfg", "cfg", "cfgpp", "cfg0s", "apg",
                "tcfg", "sag", "seg_sigma10", "oseg", "pag"]


def score_for(path: Path):
    """(mean*100, n_images) from a dpg_score.txt; None if unreadable/empty."""
    vals = []
    for line in path.read_text().splitlines():
        parts = line.rsplit(",", 2)        # path may contain commas -> split right
        if len(parts) == 3:
            try:
                vals.append(float(parts[1]))
            except ValueError:
                pass
    if not vals:
        return None
    return sum(vals) / len(vals) * 100, len(vals)


def discover(out_root: Path, tag: str):
    """Yield (method, dpg_score.txt) for outputs/dpg/<method>_<tag>/eval/."""
    found = {}
    for d in (out_root / "dpg").glob(f"*_{tag}"):
        f = d / "eval" / "dpg_score.txt"
        if f.exists():
            found[d.name[: -(len(tag) + 1)]] = f
    # canonical order first, then any extras
    ordered = [m for m in METHOD_ORDER if m in found]
    ordered += [m for m in sorted(found) if m not in ordered]
    return [(m, found[m]) for m in ordered]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("tag", help="run-tag (e.g. cosmos2_24052026)")
    ap.add_argument("--model", default="sd35", choices=["sd35", "cosmos2"])
    ap.add_argument("--out-root", default=None,
                    help="base outputs dir (default: <repo>/outputs)")
    ap.add_argument("--csv", default=None, help="also write the table as CSV")
    args = ap.parse_args()

    base = Path(args.out_root) if args.out_root else PROJECT_ROOT / "outputs"
    out_root = base / args.model if args.model != "sd35" else base

    runs = discover(out_root, args.tag)
    if not runs:
        sys.exit(f"No dpg_score.txt under {out_root}/dpg/*_{args.tag}/eval/")

    rows = []
    for m, f in runs:
        r = score_for(f)
        rows.append((m, f"{r[0]:.2f}", str(r[1])) if r else (m, "ERR", "0"))

    header = ("method", "DPG", "N")
    table = [header] + rows
    widths = [max(len(t[i]) for t in table) for i in range(3)]
    print(f"{header[0]:<{widths[0]}}  {header[1]:>{widths[1]}}  {header[2]:>{widths[2]}}")
    print("-" * (sum(widths) + 4))
    for m, s, n in rows:
        print(f"{m:<{widths[0]}}  {s:>{widths[1]}}  {n:>{widths[2]}}")
    print("\n(DPG = mean per-image score ×100; N = #images, expect 90)")

    if args.csv:
        import csv as _csv
        with open(args.csv, "w", newline="") as fh:
            w = _csv.writer(fh)
            w.writerow(["method", "dpg", "n"])
            for m, s, n in rows:
                w.writerow([m, s, n])
        print(f"wrote {args.csv}")


if __name__ == "__main__":
    main()
