"""Build a LaTeX report of image grids from existing generations.

Two families of grids:
  1. HP sweeps  — effect of varying a parameter:
       SD3.5    : outputs/sweep/<param>/<value>/<idx>.png        (1 prompt row)
       Cosmos2  : outputs/cosmos2/hp/<method>/<trial>/<idx>/samples/00000.png
  2. Best-param across methods, one grid per benchmark (rows = methods):
       GenEval  : <base>/geneval/<method>_<tag>/<NNNNN>/samples/00000.png
       DPG      : <base>/dpg/<method>_<tag>/images/<id>.png
       OneIG    : <base>/oneig/<tag>/<cat>/<method>/<NNN>.webp   (3 categories)

Makes JPG thumbnails under report/figures/ (pdflatex can't read .webp) and emits
report/report.tex. Run on the server (needs PIL + the outputs/ tree); compile
with pdflatex there.

    ~/.venv/bin/python scripts/build_report.py
"""

from __future__ import annotations

import csv
import json
import re
from collections import defaultdict
from pathlib import Path

from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUT = PROJECT_ROOT / "outputs"
REPORT = PROJECT_ROOT / "report"
FIGS = REPORT / "figures"
THUMB = 360                      # thumbnail max side (px)

TAG = {"sd35": "strat27042026", "cosmos2": "cosmos2_24052026"}
# canonical method key -> per-model on-disk token (SEG differs between models)
METHODS = ["no_cfg", "cfg", "cfgpp", "cfg0s", "apg", "tcfg", "sag", "seg", "oseg", "pag"]
DISPLAY = {"no_cfg": "no-CFG", "cfg": "CFG", "cfgpp": "CFG++", "cfg0s": "CFG-Zero*",
           "apg": "APG", "tcfg": "TCFG", "sag": "SAG", "seg": "SEG", "oseg": "OSEG",
           "pag": "PAG"}


def mtoken(m, model):
    if m == "seg":
        return "seg_sigma10_cfg3" if model == "sd35" else "seg_sigma10"
    return m


def base(model):
    return OUT if model == "sd35" else OUT / "cosmos2"


# ---- prompt lookups (column = prompt content) ------------------------------

DPG_PROMPTS = Path.home() / "ELLA" / "dpg_bench" / "prompts"
ONEIG_CSV = Path.home() / "OneIG-Benchmark" / "OneIG-Bench.csv"
ONEIG_LONG = {"object": "General_Object", "text": "Text_Rendering",
              "reasoning": "Knowledge_Reasoning"}
_ONEIG_CACHE = None


def dpg_prompt(pid):
    f = DPG_PROMPTS / f"{pid}.txt"
    return f.read_text().strip() if f.exists() else pid


def oneig_prompt(cat, pid):
    global _ONEIG_CACHE
    if _ONEIG_CACHE is None:
        _ONEIG_CACHE = {}
        if ONEIG_CSV.exists():
            with ONEIG_CSV.open() as fh:
                for row in csv.DictReader(fh):
                    _ONEIG_CACHE[(row["category"], row["id"])] = row["prompt_en"]
    return _ONEIG_CACHE.get((ONEIG_LONG.get(cat, cat), pid), pid)


def preview(s, n=60):
    s = " ".join(s.split())
    return s[:n] + "..." if len(s) > n else s


# ---- thumbnails ------------------------------------------------------------

def thumb(src: Path, rel: str, size=THUMB):
    """Make figures/<rel>.jpg from src; return rel path (for \\includegraphics)
    or None if src missing/unreadable."""
    if not src or not Path(src).exists():
        return None
    # keep '/' (subdirs) but strip dots/'='/spaces so graphicx parses .jpg right
    rel = "/".join(re.sub(r"[^A-Za-z0-9_-]", "_", seg) for seg in rel.split("/"))
    dst = FIGS / f"{rel}.jpg"
    if not dst.exists():
        dst.parent.mkdir(parents=True, exist_ok=True)
        try:
            im = Image.open(src).convert("RGB")
            im.thumbnail((size, size), Image.LANCZOS)
            im.save(dst, "JPEG", quality=85)
        except Exception as e:                       # noqa: BLE001
            print(f"thumb FAIL {src}: {e}")
            return None
    return f"figures/{rel}.jpg"


# ---- LaTeX helpers ---------------------------------------------------------

def tex_escape(s: str) -> str:
    for a, b in [("\\", r"\textbackslash{}"), ("_", r"\_"), ("%", r"\%"),
                 ("&", r"\&"), ("#", r"\#"), ("$", r"\$"), ("{", r"\{"),
                 ("}", r"\}"), ("^", r"\^{}"), ("~", r"\~{}")]:
        s = s.replace(a, b)
    return s


def grid_figure(caption, col_labels, row_labels, cells, label_w=1.5, placement="H",
                header_mode="auto", legend=None):
    """cells[r][c] = rel image path or None. Returns a LaTeX figure string.
    Image width is sized so tall (10-row) grids still fit a portrait page.
    header_mode: "auto" rotates long labels vertical; "wrap" sets a wrapped
    text box above each column (for prompt previews). `legend` = [(key, full
    text)] is printed as a footnote paragraph under the figure."""
    ncol = len(col_labels)
    cap = 1.7 if len(row_labels) > 3 else 2.1     # tall grids need smaller cells
    imgw = min(cap, 16.0 / (ncol + 0.8))
    colspec = "@{}p{%.2fcm}@{\\hspace{1mm}}%s@{}" % (label_w, "".join("c" for _ in col_labels))
    lines = [r"\begin{figure}[%s]\centering" % placement, r"\footnotesize",
             r"\setlength{\tabcolsep}{1.2pt}\renewcommand{\arraystretch}{0.4}",
             r"\begin{tabular}{%s}" % colspec]
    # header row
    if header_mode == "wrap":
        hdr = [""] + [r"\parbox[b]{%.2fcm}{\tiny\raggedright %s\strut}" % (imgw, tex_escape(c))
                      for c in col_labels]
    else:
        rot = max((len(c) for c in col_labels), default=0) > 6
        cell = (r"\rotatebox{90}{\tiny %s}" if rot else r"\tiny %s")
        hdr = [""] + [cell % tex_escape(c) for c in col_labels]
    lines.append(" & ".join(hdr) + r" \\[2pt]")
    for r, rlab in enumerate(row_labels):
        cellsr = [r"\tiny %s" % tex_escape(rlab)]
        for c in range(ncol):
            p = cells[r][c]
            if p:
                cellsr.append(r"\includegraphics[width=%.2fcm,height=%.2fcm]{%s}"
                              % (imgw, imgw, p))
            else:
                cellsr.append(r"\rule{%.2fcm}{0cm}" % imgw)
        lines.append(" & ".join(cellsr) + r" \\")
    lines += [r"\end{tabular}", r"\caption{%s}" % caption, r"\end{figure}"]
    if legend:
        leg = "\\quad ".join(r"\textbf{%s:} %s" % (tex_escape(k), tex_escape(" ".join(t.split())))
                             for k, t in legend)
        lines.append(r"{\footnotesize\noindent\textbf{Prompts.}~%s\par}" % leg)
    return "\n".join(lines)


# ---- selection -------------------------------------------------------------

def geneval_picks(model):
    """1 prompt per distinct tag (category), from the cfg run. Returns
    [(idx, tag)] using idx present across methods (first 30)."""
    run = base(model) / "geneval" / f"{mtoken('cfg', model)}_{TAG[model]}"
    picks, seen = [], set()
    for i in range(60):
        idx = f"{i:05d}"
        meta = run / idx / "metadata.jsonl"
        if not meta.exists():
            continue
        tag = json.loads(meta.read_text().splitlines()[0]).get("tag", idx)
        if tag not in seen:
            seen.add(tag)
            picks.append((idx, tag))
    return picks


def dpg_picks(model, n=5):
    run = base(model) / "dpg" / f"{mtoken('cfg', model)}_{TAG[model]}" / "images"
    ids = sorted((p.stem for p in run.glob("*.png")), key=lambda s: (len(s), s))
    return ids[:n]


def oneig_picks(model, cat, n=4):
    root = base(model) / "oneig" / TAG[model] / cat
    sets = []
    for m in METHODS:
        d = root / mtoken(m, model)
        if d.exists():
            sets.append({p.stem for p in d.glob("*.webp")})
    common = sorted(set.intersection(*sets)) if sets else []
    return common[:n]


# ---- quantitative metrics (mirror scripts/summarize_*.py) ------------------

GCATS = ["single_object", "two_object", "counting", "colors", "position", "color_attr"]
GCAT_SHORT = {"single_object": "single", "two_object": "two", "counting": "count",
              "colors": "colors", "position": "pos", "color_attr": "attr"}
ONEIG_FAM = {"object": ("alignment_score_EN_", "alignment"),
             "text": ("text_score_EN_", "text score"),
             "reason": ("reasoning_score_EN_", "reasoning")}


def metric_geneval(model):
    res = {}
    for m in METHODS:
        f = base(model) / "geneval" / f"{mtoken(m, model)}_{TAG[model]}" / "results.jsonl"
        if not f.exists():
            continue
        by = defaultdict(lambda: [0, 0])
        for line in f.read_text().splitlines():
            if not line.strip():
                continue
            d = json.loads(line)
            by[d.get("tag")][0] += bool(d.get("correct"))
            by[d.get("tag")][1] += 1
        cat = {t: (v[0] / v[1] if v[1] else 0.0) for t, v in by.items()}
        res[m] = {"cat": cat, "overall": sum(cat.values()) / len(cat) if cat else 0.0}
    return res


def metric_dpg(model):
    res = {}
    for m in METHODS:
        f = base(model) / "dpg" / f"{mtoken(m, model)}_{TAG[model]}" / "eval" / "dpg_score.txt"
        if not f.exists():
            continue
        vals = []
        for line in f.read_text().splitlines():
            p = line.rsplit(",", 2)
            if len(p) == 3:
                try:
                    vals.append(float(p[1]))
                except ValueError:
                    pass
        if vals:
            res[m] = sum(vals) / len(vals) * 100
    return res


def metric_oneig(model):
    res = {}
    for m in METHODS:
        d = base(model) / "oneig" / TAG[model] / "eval" / mtoken(m, model)
        if not d.is_dir():
            continue
        rec = {}
        for fam, (pref, col) in ONEIG_FAM.items():
            cands = sorted(d.glob(f"{pref}*.csv"), key=lambda p: p.stat().st_mtime)
            if not cands:
                continue
            for r in csv.DictReader(cands[-1].open()):
                if (r.get("") or "").strip() == mtoken(m, model):
                    try:
                        rec[fam] = float(r[col])
                    except (ValueError, KeyError):
                        pass
                    break
        if rec:
            res[m] = rec
    return res


def latex_geneval_table(model, res):
    head = " & ".join(["Method"] + [GCAT_SHORT[c] for c in GCATS] + [r"\textbf{Overall}"])
    lines = [r"\begin{table}[H]\centering\small",
             r"\setlength{\tabcolsep}{4pt}",
             r"\begin{tabular}{l%s}" % ("r" * (len(GCATS) + 1)), r"\hline",
             head + r" \\", r"\hline"]
    for m in METHODS:
        if m not in res:
            continue
        r = res[m]
        cells = [DISPLAY[m]] + [f"{r['cat'].get(c, 0):.3f}" for c in GCATS]
        cells.append(r"\textbf{%.3f}" % r["overall"])
        lines.append(" & ".join(cells) + r" \\")
    lines += [r"\hline", r"\end{tabular}",
              r"\caption{GenEval --- %s. Per-category accuracy; Overall = macro average over tasks.}" % model,
              r"\end{table}"]
    return "\n".join(lines)


def latex_other_table(model, dpg, oneig):
    head = " & ".join(["Method", r"DPG", r"OneIG-obj", r"OneIG-text", r"OneIG-reason"])
    lines = [r"\begin{table}[H]\centering\small",
             r"\setlength{\tabcolsep}{6pt}",
             r"\begin{tabular}{lrrrr}", r"\hline", head + r" \\", r"\hline"]
    for m in METHODS:
        o = oneig.get(m, {})
        cells = [DISPLAY[m],
                 f"{dpg[m]:.2f}" if m in dpg else "--",
                 f"{o['object']:.3f}" if "object" in o else "--",
                 f"{o['text']:.3f}" if "text" in o else "--",
                 f"{o['reason']:.3f}" if "reason" in o else "--"]
        lines.append(" & ".join(cells) + r" \\")
    lines += [r"\hline", r"\end{tabular}",
              r"\caption{DPG-Bench (mean$\times$100) and OneIG (0--1) --- %s.}" % model,
              r"\end{table}"]
    return "\n".join(lines)


# ---- grid builders ---------------------------------------------------------

def build_geneval(model):
    picks = geneval_picks(model)
    if not picks:
        return ""
    cols = [t for _, t in picks]
    cells = []
    for m in METHODS:
        run = base(model) / "geneval" / f"{mtoken(m, model)}_{TAG[model]}"
        row = [thumb(run / idx / "samples" / "00000.png",
                     f"best/{model}/geneval/{m}_{idx}") for idx, _ in picks]
        cells.append(row)
    return grid_figure(f"GenEval --- {model}: rows = methods, columns = task categories.",
                       cols, [DISPLAY[m] for m in METHODS], cells)


def build_dpg(model):
    ids = dpg_picks(model)
    if not ids:
        return ""
    keys = [f"P{i + 1}" for i in range(len(ids))]
    prompts = [dpg_prompt(i) for i in ids]
    col_labels = [f"{k}: {preview(p)}" for k, p in zip(keys, prompts)]
    legend = list(zip(keys, prompts))
    cells = []
    for m in METHODS:
        run = base(model) / "dpg" / f"{mtoken(m, model)}_{TAG[model]}" / "images"
        row = [thumb(run / f"{i}.png", f"best/{model}/dpg/{m}_{i}") for i in ids]
        cells.append(row)
    return grid_figure(f"DPG-Bench --- {model}: rows = methods, columns = sample prompts.",
                       col_labels, [DISPLAY[m] for m in METHODS], cells,
                       header_mode="wrap", legend=legend)


def build_oneig(model, cat):
    nums = oneig_picks(model, cat)
    if not nums:
        return ""
    keys = [f"P{i + 1}" for i in range(len(nums))]
    prompts = [oneig_prompt(cat, n) for n in nums]
    col_labels = [f"{k}: {preview(p)}" for k, p in zip(keys, prompts)]
    legend = list(zip(keys, prompts))
    cells = []
    for m in METHODS:
        d = base(model) / "oneig" / TAG[model] / cat / mtoken(m, model)
        row = [thumb(d / f"{n}.webp", f"best/{model}/oneig_{cat}/{m}_{n}") for n in nums]
        cells.append(row)
    return grid_figure(f"OneIG / {cat} --- {model}: rows = methods, columns = sample prompts.",
                       col_labels, [DISPLAY[m] for m in METHODS], cells,
                       header_mode="wrap", legend=legend)


def build_sd_sweeps():
    SWEEPS = [
        ("cfgpp_scale", "CFG++: guidance scale"),
        ("cfg0s_scale", "CFG-Zero*: guidance scale"),
        ("cfg0s_zsteps", "CFG-Zero*: zero-init steps"),
        ("oseg_sscale", "OSEG: SEG scale"),
        ("oseg_oscale", "OSEG: orthogonal scale"),
        ("pag_scale", "PAG: guidance scale"),
        ("tcfg_scale", "TCFG: guidance scale"),
        ("tcfg_rank", "TCFG: rank"),
    ]
    out = []
    for param, cap in SWEEPS:
        root = OUT / "sweep" / param
        if not root.is_dir():
            continue
        vals = sorted(p for p in root.iterdir() if p.is_dir())
        cols = [v.name.split("=", 1)[1] for v in vals]
        # two representative prompts (rows)
        idxs = sorted({p.stem for p in vals[0].glob("*.png")})[:2]
        rows, rlabels = [], []
        for idx in idxs:
            row = [thumb(v / f"{idx}.png", f"sweep/sd35/{param}_{v.name}_{idx}") for v in vals]
            rows.append(row)
            txt = vals[0] / f"{idx}.txt"
            rlabels.append((txt.read_text().strip()[:40]) if txt.exists() else idx)
        out.append(grid_figure(f"SD3.5 HP sweep --- {cap} (columns = values).",
                               cols, rlabels, rows, placement="H"))
    return "\n\n".join(out)


ABBR = {"seg_applied_layers": "L", "pag_applied_layers": "L", "seg_blur_sigma": "sig",
        "zero_steps": "z", "apg_momentum": "mom", "guidance": "g", "segs": "s",
        "pags": "s", "cfgpp_w": "w"}


def shorten_trials(trials):
    """Compact, readable column labels: keep only keys that vary across trials and
    abbreviate verbose key names."""
    parsed = [dict(t.split("=", 1) for t in name.split(".") if "=" in t) for name in trials]
    keys = set().union(*[p.keys() for p in parsed]) if parsed else set()
    varying = sorted(k for k in keys if len({p.get(k) for p in parsed}) > 1)
    labels = []
    for p in parsed:
        labels.append(", ".join(f"{ABBR.get(k, k)}={p.get(k, '?')}" for k in varying) or "default")
    return labels


def build_cosmos_sweeps():
    root = OUT / "cosmos2" / "hp"
    if not root.is_dir():
        return ""
    out = []
    for m in METHODS:
        mdir = root / ("seg" if m == "seg" else m)
        if not mdir.is_dir():
            continue
        trials = sorted(p.name for p in mdir.iterdir() if p.is_dir())
        if len(trials) < 2:                          # nothing to compare
            continue
        labels = shorten_trials(trials)
        idx = "00000"
        row = [thumb(mdir / t / idx / "samples" / "00000.png",
                     f"sweep/cosmos2/{m}_{i}") for i, t in enumerate(trials)]
        out.append(grid_figure(f"Cosmos2 HP sweep --- {DISPLAY.get(m, m)} (columns = trials).",
                               labels, [DISPLAY.get(m, m)], [row], placement="H"))
    return "\n\n".join(out)


# ---- main ------------------------------------------------------------------

DOC_HEAD = r"""\documentclass[a4paper,10pt]{article}
\usepackage[margin=1.2cm]{geometry}
\usepackage[utf8]{inputenc}
\usepackage[T1]{fontenc}
\usepackage{graphicx}
\usepackage{array}
\usepackage{caption}
\usepackage{rotating}
\usepackage{float}
\captionsetup{font=small,labelfont=bf}
\graphicspath{{./}}
\setlength{\parindent}{0pt}
\title{Guidance methods --- qualitative comparison\\\large SD3.5-medium \& Cosmos-Predict2-2B}
\author{}\date{}
\begin{document}
\maketitle
This report collects qualitative image grids. Part~I shows the effect of the
swept hyper-parameters per method; Part~II compares all methods at their best
parameters on samples from each benchmark.
"""


def main():
    FIGS.mkdir(parents=True, exist_ok=True)
    parts = [DOC_HEAD]

    parts.append(r"\section{Quantitative results}")
    for model in ("sd35", "cosmos2"):
        parts.append(r"\subsection{%s}" % {"sd35": "SD3.5", "cosmos2": "Cosmos2"}[model])
        parts.append(latex_geneval_table(model, metric_geneval(model)))
        parts.append(latex_other_table(model, metric_dpg(model), metric_oneig(model)))

    parts.append(r"\clearpage\section{HP sweeps}")
    parts.append(r"\subsection{SD3.5}")
    parts.append(build_sd_sweeps())
    parts.append(r"\clearpage\subsection{Cosmos2}")
    parts.append(build_cosmos_sweeps())

    parts.append(r"\section{Best-parameter comparison}")
    for model in ("sd35", "cosmos2"):
        parts.append(r"\clearpage\subsection{%s}" % {"sd35": "SD3.5", "cosmos2": "Cosmos2"}[model])
        parts.append(build_geneval(model))
        parts.append(build_dpg(model))
        for cat in ("object", "text", "reasoning"):
            parts.append(build_oneig(model, cat))

    parts.append(r"\end{document}")
    tex = "\n\n".join(p for p in parts if p)
    (REPORT / "report.tex").write_text(tex)
    n = len(list(FIGS.rglob("*.jpg")))
    print(f"wrote {REPORT/'report.tex'}  ({n} thumbnails)")


if __name__ == "__main__":
    main()
