"""Aggregate GenEval ``results.jsonl`` across methods into one comparison table.

Mirrors the upstream geneval ``summary_scores.py`` metric definitions:
  * per-task score = mean(correct) within a tag
  * Overall        = mean over the per-task scores  (macro-average — the
                     headline "Overall score (avg. over tasks)")
  * Img%  (micro)  = mean(correct) over all images
  * Prompt%        = fraction of prompts (grouped by ``metadata``) with at
                     least one correct image

For every ``outputs[/<model>]/geneval/<method>_<tag>/results.jsonl`` it prints
one row; columns are the tags (canonical order) followed by Overall.

Usage:
    scripts/summarize_geneval.py <tag> [--model sd35|cosmos2]
                                 [--out-root DIR] [--csv FILE] [--pct] [--extra]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Canonical task order (GenEval ships prompts in this order). Unknown tags are
# appended after these in first-seen order.
CANON_TAGS = ["single_object", "two_object", "counting",
              "colors", "position", "color_attr"]
SHORT = {"single_object": "single", "two_object": "two_obj", "counting": "count",
         "colors": "colors", "position": "posit", "color_attr": "col_attr",
         "Overall": "OVERALL", "Img": "img", "Prompt": "prompt"}


def load_results(path: Path):
    """Return list of (tag, correct, metadata-key) from a results.jsonl."""
    rows = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            rows.append((d.get("tag"), bool(d.get("correct")),
                         json.dumps(d.get("metadata"), sort_keys=True)))
    return rows


def summarize(rows):
    """Per-tag means + macro / micro / prompt scores for one method."""
    by_tag = {}                       # tag -> [n_correct, n_total]
    by_prompt = {}                    # metadata-key -> any_correct
    n_correct = n_total = 0
    for tag, correct, meta in rows:
        c = by_tag.setdefault(tag, [0, 0])
        c[0] += correct
        c[1] += 1
        by_prompt[meta] = by_prompt.get(meta, False) or correct
        n_correct += correct
        n_total += 1
    tag_score = {t: (c[0] / c[1] if c[1] else 0.0) for t, c in by_tag.items()}
    macro = sum(tag_score.values()) / len(tag_score) if tag_score else 0.0
    return {"tags": tag_score,
            "overall": macro,
            "img": n_correct / n_total if n_total else 0.0,
            "prompt": sum(by_prompt.values()) / len(by_prompt) if by_prompt else 0.0,
            "n_img": n_total, "n_prompt": len(by_prompt)}


def discover(out_root: Path, tag: str):
    """Yield (method, results_path) for outputs/geneval/<method>_<tag>/."""
    for d in sorted((out_root / "geneval").glob(f"*_{tag}")):
        f = d / "results.jsonl"
        if f.exists():
            yield d.name[: -(len(tag) + 1)], f


def order_tags(stats):
    seen = []
    for s in stats.values():
        for t in s["tags"]:
            if t not in seen:
                seen.append(t)
    canon = [t for t in CANON_TAGS if t in seen]
    return canon + [t for t in seen if t not in canon]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("tag", help="run-tag suffix (e.g. cosmos2_24052026)")
    ap.add_argument("--model", default="sd35", choices=["sd35", "cosmos2"])
    ap.add_argument("--out-root", default=None,
                    help="base outputs dir (default: <repo>/outputs)")
    ap.add_argument("--csv", default=None, help="also write the table as CSV")
    ap.add_argument("--pct", action="store_true", help="print percentages")
    ap.add_argument("--extra", action="store_true",
                    help="add Img (micro) and Prompt columns")
    args = ap.parse_args()

    base = Path(args.out_root) if args.out_root else PROJECT_ROOT / "outputs"
    out_root = base / args.model if args.model != "sd35" else base

    stats = {m: summarize(load_results(f)) for m, f in discover(out_root, args.tag)}
    if not stats:
        sys.exit(f"No runs match {out_root}/geneval/*_{args.tag}")

    cols = order_tags(stats) + ["Overall"] + (["Img", "Prompt"] if args.extra else [])
    fmt = (lambda v: f"{v * 100:.1f}") if args.pct else (lambda v: f"{v:.3f}")

    def value(s, col):
        return {"Overall": s["overall"], "Img": s["img"],
                "Prompt": s["prompt"]}.get(col, s["tags"].get(col, 0.0))

    header = ["method"] + [SHORT.get(c, c) for c in cols]
    body = [[m] + [fmt(value(s, c)) for c in cols] for m, s in stats.items()]
    table = [header] + body
    widths = [max(len(r[i]) for r in table) for i in range(len(header))]

    def line(r):
        return "  ".join(c.rjust(widths[i]) if i else c.ljust(widths[i])
                         for i, c in enumerate(r))

    print(line(header))
    print("-" * (sum(widths) + 2 * (len(widths) - 1)))
    for r in body:
        print(line(r))
    print("\n(Overall = macro avg over tasks, matching upstream "
          '"Overall score (avg. over tasks)")')

    if args.csv:
        import csv as _csv
        with open(args.csv, "w", newline="") as fh:
            w = _csv.writer(fh)
            w.writerow(["method"] + cols)
            for m, s in stats.items():
                w.writerow([m] + [f"{value(s, c):.5f}" for c in cols])
        print(f"wrote {args.csv}")


if __name__ == "__main__":
    main()
