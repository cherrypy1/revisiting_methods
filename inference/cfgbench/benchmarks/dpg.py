"""DPG-Bench adapter.

prompts: ``ELLA/dpg_bench/prompts/<id>.txt`` (long paragraphs), single category "dpg".
evaluate: bridge → flat ``<id>.png`` dir, run ``compute_dpg_bench.py`` (mPLUG, single-process),
          parse ``dpg_score.txt`` per-image lines ``<path>, <s>, <s>`` (col[1]).
summarize: mean per-image score (×100 in display).
"""

from __future__ import annotations

import os
from pathlib import Path

from .base import BenchmarkAdapter, PromptSpec
from ._util import env_path, link_or_copy, run


class DPGAdapter(BenchmarkAdapter):
    name = "dpg"
    eval_needs_gpu = True
    default_samples_per_prompt = 1

    def prompts(self):
        d = env_path("DPG_PROMPTS", Path.home() / "ELLA" / "dpg_bench" / "prompts")
        return [PromptSpec(id=f.stem, text=f.read_text().strip(), category="dpg")
                for f in sorted(d.glob("*.txt"))]

    def evaluate(self, items, workdir):
        workdir = Path(workdir)
        imgs = workdir / "imgs"
        imgs.mkdir(parents=True, exist_ok=True)
        by_prompt = {}
        for it in items:
            if it.sample_idx == 0:  # pic-num=1: one image per prompt
                link_or_copy(it.image_path, imgs / f"{it.prompt.id}.png")
                by_prompt[it.prompt.id] = it.item_id

        root = env_path("DPG_ROOT", Path.home() / "ELLA")
        compute = root / "dpg_bench" / "compute_dpg_bench.py"
        res = workdir / "dpg_score.txt"
        run(["accelerate", "launch", "--num_machines", "1", "--num_processes", "1",
             "--mixed_precision", "fp16", "--main_process_port", os.environ.get("DPG_PORT", "29500"),
             compute, "--image-root-path", imgs, "--resolution", "1024", "--pic-num", "1",
             "--vqa-model", "mplug", "--res-path", res], cwd=root)

        metrics = {}
        for line in res.read_text().splitlines():
            parts = line.rsplit(",", 2)
            if len(parts) != 3:
                continue
            try:
                score = float(parts[1])
            except ValueError:
                continue
            iid = by_prompt.get(Path(parts[0].strip()).stem)
            if iid:
                metrics[iid] = {"dpg": score}
        return metrics

    def summarize(self, metrics, prompts):
        vals = [m["dpg"] for m in metrics.values() if "dpg" in m]
        mean = sum(vals) / len(vals) if vals else 0.0
        cell = {"dpg": mean, "dpg_x100": mean * 100, "n": len(vals)}
        return {"by_category": {"dpg": cell}, "overall": cell, "n": len(vals)}
