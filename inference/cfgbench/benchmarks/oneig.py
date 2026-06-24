"""OneIG-Benchmark adapter (categories: General_Object / Text_Rendering / Knowledge_Reasoning).

prompts: ``OneIG-Bench.csv`` (category,id,prompt_en) → short categories object/text/reasoning.
evaluate: bridge → ``<imgs>/<short>/<tag>/<id>.webp``, run the 3 score modules in
          a private OneIG worktree view, then read freshly-written CSVs from its
          private results dir.
          - aggregate ``<fam>_score_EN_*.csv``        → per-category score (parity source)
          - per-prompt ``<fam>_prompt_score_EN_*.csv`` → per-item metric
summarize: per-category aggregate (object=alignment, text="text score" composite, reason=reasoning).

NOTE OneIG hardcodes output to ``./results`` (+ shared tmp dirs). cfgbench avoids
cross-run races by creating a private per-eval OneIG root view under ``workdir``:
benchmark code/data are symlinked from $ONEIG_ROOT, while ``results/`` and tmp
dirs are local to the current eval shard.
"""

from __future__ import annotations

import ast
import csv
import os
import subprocess
import sys
import time
from pathlib import Path

from .base import BenchmarkAdapter, PromptSpec
from ._util import env_path, run, to_webp

ONEIG_ROOT = env_path("ONEIG_ROOT", Path.home() / "OneIG-Benchmark")
CAT_SHORT = {"General_Object": "object", "Text_Rendering": "text",
             "Knowledge_Reasoning": "reasoning"}
# short -> (aggregate file prefix, aggregate score column, per-prompt file prefix)
FAMILY = {
    "object":    ("alignment_score_EN_", "alignment", "alignment_prompt_score_EN_"),
    "text":      ("text_score_EN_",      "text score", "text_prompt_score_EN_"),
    "reasoning": ("reasoning_score_EN_", "reasoning", "reasoning_prompt_score_EN_"),
}
_MODULE = {"object": "scripts.alignment.alignment_score",
           "text": "scripts.text.text_score",
           "reasoning": "scripts.reasoning.reasoning_score"}
CAT_ORDER = ["reasoning", "object", "text"]


def _agg_value(path: Path, tag: str, col: str):
    with path.open() as f:
        for r in csv.DictReader(f):
            if (r.get("") or "").strip() == tag:
                try:
                    return float(r.get(col))
                except (TypeError, ValueError):
                    return None
    return None


def _prompt_values(path: Path, tag: str) -> dict:
    out = {}
    with path.open() as f:
        for r in csv.DictReader(f):
            pid = (r.get("") or "").strip()
            if pid:
                out[pid] = r.get(tag)
    return out


def _candidates_after(results: Path, prefix: str, t0: float):
    cands = [p for p in results.glob(prefix + "*.csv") if p.stat().st_mtime >= t0 - 1]
    return sorted(cands, key=lambda p: p.stat().st_mtime, reverse=True)


def _oneig_env() -> dict:
    env = os.environ.copy()
    cache = env.get("ONEIG_HF_HOME") or env.get("CFG_HF_HOME")
    if cache:
        hf_home = Path(cache).expanduser()
        hf_home.mkdir(parents=True, exist_ok=True)
        env["HF_HOME"] = str(hf_home)
        env["HUGGINGFACE_HUB_CACHE"] = str(hf_home / "hub")
    env.setdefault("HF_HUB_DISABLE_XET", "1")
    return env


def _cat_order(cats) -> list[str]:
    return [cat for cat in CAT_ORDER if cat in cats]


def _private_oneig_root(workdir: Path) -> Path:
    private = workdir / "oneig_root"
    private.mkdir(parents=True, exist_ok=True)
    (private / "results").mkdir(exist_ok=True)

    for src in ONEIG_ROOT.iterdir():
        name = src.name
        if name == "results" or name.startswith("tmp"):
            continue
        dst = private / name
        if dst.exists() or dst.is_symlink():
            continue
        try:
            os.symlink(src, dst, target_is_directory=src.is_dir())
        except FileExistsError:
            pass
    return private


class OneIGAdapter(BenchmarkAdapter):
    name = "oneig"
    eval_needs_gpu = True
    default_samples_per_prompt = 1

    def __init__(self):
        self._agg = {}  # short -> aggregate score (filled in evaluate, used by summarize)

    def prompts(self):
        path = env_path("ONEIG_CSV", ONEIG_ROOT / "OneIG-Bench.csv")
        out = []
        with path.open() as f:
            for row in csv.DictReader(f):
                short = CAT_SHORT.get(row.get("category"))
                if short:
                    out.append(PromptSpec(id=row["id"], text=row["prompt_en"],
                                          category=short, meta={"category": row.get("category")}))
        return out

    def evaluate(self, items, workdir):
        workdir = Path(workdir)
        imgs = workdir / "imgs"
        tag = self._tag(items)
        cats = _cat_order({it.prompt.category for it in items} & set(FAMILY))

        for it in items:
            if it.sample_idx == 0 and it.prompt.category in FAMILY:
                to_webp(it.image_path, imgs / it.prompt.category / tag / f"{it.prompt.id}.webp")

        oneig_root = _private_oneig_root(workdir)
        results = oneig_root / "results"
        env = _oneig_env()
        common = ["--mode", "EN", "--model_names", tag, "--image_grid", "1,1"]
        metrics = {}
        # map (category, prompt_id) -> item_id for sample 0
        lut = {(it.prompt.category, it.prompt.id): it.item_id
               for it in items if it.sample_idx == 0}

        for short in cats:
            agg_pref, agg_col, prm_pref = FAMILY[short]
            dirname = imgs if short == "object" else imgs / short
            extra = ["--class_items", "object"] if short == "object" else []
            t0 = time.time()
            try:
                run([sys.executable, "-m", _MODULE[short], "--image_dirname", dirname, *extra, *common],
                    cwd=oneig_root, env=env)
            except subprocess.CalledProcessError as e:
                hint = (
                    "OneIG scorer failed; inspect the worker log above for the real "
                    "import/runtime error. If it mentions GLIBCXX/libstdc++, load "
                    "gnu14/14.1 or prepend its libstdc++ to LD_LIBRARY_PATH. If it "
                    "mentions HuggingFace 'Disk quota exceeded', set ONEIG_HF_HOME or "
                    "CFG_HF_HOME to a scratch/cache directory and delete any partial "
                    "failed snapshot from the old cache."
                )
                raise RuntimeError(hint) from e

            for agg_csv in _candidates_after(results, agg_pref, t0):
                v = _agg_value(agg_csv, tag, agg_col)
                if v is not None:
                    self._agg[short] = v
                    break

            for prm_csv in _candidates_after(results, prm_pref, t0):
                values = _prompt_values(prm_csv, tag)
                if not any(v not in (None, "") for v in values.values()):
                    continue
                for pid, raw in values.items():
                    # alignment scorer prefixes object ids with the class name
                    # ("object_000"); strip it back to our prompt id ("000").
                    if short == "object" and pid.startswith("object_"):
                        pid = pid[len("object_"):]
                    iid = lut.get((short, pid))
                    if iid:
                        metrics[iid] = self._parse_item(short, raw)
                break
        return metrics

    @staticmethod
    def _tag(items) -> str:
        first = items[0].item_id.split("/")
        return f"{first[0]}_{first[1]}"  # model_config — unique, readable

    @staticmethod
    def _parse_item(short: str, raw) -> dict:
        if short == "text":
            try:
                ed, cr, wac = ast.literal_eval(raw)
                return {"ED": float(ed), "CR": float(cr), "WAC": float(wac)}
            except Exception:
                return {}
        try:
            return {short if short != "object" else "alignment": float(raw)}
        except (TypeError, ValueError):
            return {}

    def summarize(self, metrics, prompts):
        from collections import Counter
        cnt = Counter(p.category for p in prompts)
        cats = {}
        for short in FAMILY:
            if short in self._agg:
                cats[short] = {"score": self._agg[short], "n": cnt.get(short, 0)}
        overall = ({"score": sum(c["score"] for c in cats.values()) / len(cats)}
                   if cats else {})
        return {"by_category": cats, "overall": overall, "n": sum(cnt.values())}
