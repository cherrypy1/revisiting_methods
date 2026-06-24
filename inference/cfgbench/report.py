"""Campaign roll-up + report (markdown + CSV) from the uniform ``_summary.json`` tree.

Reads the per-bench / per-category summaries the worker already wrote, builds run-level
summaries, and renders:
  * an overall table (rows = model/config, cols = ``<bench>.<metric>``)
  * a per-bench category breakdown table (primary metric per category)

A LaTeX image-grid report (port of the old ``scripts/build_report.py``) is a follow-on; the
uniform layout (``<model>/<config>/<bench>/<category>/<prompt>/sample_*.png``) makes it easy.
"""

from __future__ import annotations

import csv as _csv
import io
from collections import defaultdict
from pathlib import Path

from .config import CampaignSpec, load_campaign
from .core import manifest as M
from .core.layout import CampaignPaths, read_json, write_json_atomic

# preferred scalar to show for a category cell, in priority order
_PRIMARY = ["correct", "score", "dpg_x100", "alignment", "reasoning", "dpg"]


def write_run_summaries(paths: CampaignPaths, refs: list) -> None:
    """Aggregate each (model, config)'s per-bench overalls into a run _summary.json."""
    runs = defaultdict(set)
    for r in refs:
        runs[(r.model, r.config)].add(r.bench)
    for (m, c), benches in sorted(runs.items()):
        agg = {}
        for b in sorted(benches):
            s = read_json(paths.summary_bench(m, c, b))
            if s:
                agg[b] = s.get("overall", {})
        write_run_summary(paths, m, c, agg)


def write_run_summary(paths, m, c, agg):
    write_json_atomic(paths.summary_run(m, c), {"model": m, "config": c, "benches": agg})


def _fmt(v):
    if isinstance(v, bool):
        return str(v)
    if isinstance(v, (int, float)):
        return f"{v:.4f}"
    return "-" if v is None else str(v)


def _md_table(headers, body) -> str:
    out = ["| " + " | ".join(headers) + " |",
           "|" + "|".join("---" for _ in headers) + "|"]
    for row in body:
        out.append("| " + " | ".join(row) + " |")
    return "\n".join(out)


def _primary(cell):
    if not isinstance(cell, dict):
        return cell
    for k in _PRIMARY:
        if k in cell:
            return cell[k]
    for k, v in cell.items():
        if k != "n" and isinstance(v, (int, float)) and not isinstance(v, bool):
            return v
    return None


def _overall_table(paths: CampaignPaths):
    rows = defaultdict(dict)
    cols = []
    for f in sorted(paths.root.glob("*/*/_summary.json")):
        d = read_json(f) or {}
        if "model" not in d:
            continue
        key = (d["model"], d["config"])
        for bench, overall in (d.get("benches") or {}).items():
            for k, v in overall.items():
                col = f"{bench}.{k}"
                if col not in cols:
                    cols.append(col)
                rows[key][col] = v
    return rows, cols


def _bench_cat_tables(paths: CampaignPaths):
    tables = defaultdict(lambda: ([], defaultdict(dict)))  # bench -> (cats, rows)
    for f in sorted(paths.root.glob("*/*/*/_summary.json")):
        rel = f.relative_to(paths.root).parts
        if len(rel) != 4:  # model/config/bench/_summary.json
            continue
        m, c, bench, _ = rel
        d = read_json(f) or {}
        cats, rows = tables[bench]
        for cat, cell in (d.get("by_category") or {}).items():
            if cat not in cats:
                cats.append(cat)
            rows[(m, c)][cat] = _primary(cell)
    return tables


def render(paths: CampaignPaths) -> str:
    rows, cols = _overall_table(paths)
    parts = [f"# cfgbench report — {paths.root.name}\n"]

    parts.append("## Overall\n")
    if rows:
        headers = ["model", "config"] + cols
        body = [[m, c] + [_fmt(rows[(m, c)].get(col)) for col in cols]
                for (m, c) in sorted(rows)]
        parts.append(_md_table(headers, body))
    else:
        parts.append("_no summaries found_")
    parts.append("")

    for bench, (cats, brows) in sorted(_bench_cat_tables(paths).items()):
        parts.append(f"## {bench} — by category\n")
        headers = ["model", "config"] + cats
        body = [[m, c] + [_fmt(brows[(m, c)].get(cat)) for cat in cats]
                for (m, c) in sorted(brows)]
        parts.append(_md_table(headers, body))
        parts.append("")
    return "\n".join(parts)


def _write_csv(paths: CampaignPaths) -> Path:
    rows, cols = _overall_table(paths)
    out = paths.root / "report.csv"
    with out.open("w", newline="") as f:
        w = _csv.writer(f)
        w.writerow(["model", "config"] + cols)
        for (m, c) in sorted(rows):
            w.writerow([m, c] + [rows[(m, c)].get(col, "") for col in cols])
    return out


def build(spec: CampaignSpec):
    """Write run summaries + report.md + report.csv. Returns (md_text, md_path)."""
    paths = CampaignPaths(spec.out_root)
    refs = M.load_manifest(paths)
    if not refs:
        from .benchmarks.registry import get_benchmark
        refs = [it.ref for it in M.expand(spec, lambda b: get_benchmark(b).prompts())]
    write_run_summaries(paths, refs)
    md = render(paths)
    md_path = paths.root / "report.md"
    md_path.write_text(md)
    _write_csv(paths)
    return md, md_path
