"""GenEval adapter.

prompts: ``evaluation_metadata.jsonl`` (one ``{prompt, tag, ...}`` per line).
evaluate: bridge uniform layout → GenEval layout (``<pid>/samples/<idx>.png`` + metadata.jsonl),
          run Mask2Former+CLIP scorer, parse ``results.jsonl`` (per-image ``filename``/``correct``/``tag``).
summarize: per-tag mean(correct); Overall = macro (mean of per-tag means), matching upstream.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

from .base import BenchmarkAdapter, GenItem, PromptSpec
from ._util import env_path, link_or_copy, run

GENEVAL_ROOT = env_path("GENEVAL_ROOT", Path.home() / "geneval-bench")
_DEFAULT_MM = ("mmdetection/configs/mask2former/"
               "mask2former_swin-s-p4-w7-224_8xb2-lsj-50e_coco.py")


class GenEvalAdapter(BenchmarkAdapter):
    name = "geneval"
    eval_needs_gpu = True
    default_samples_per_prompt = 4

    def prompts(self):
        path = env_path("GENEVAL_PROMPTS", GENEVAL_ROOT / "prompts" / "evaluation_metadata.jsonl")
        out = []
        for i, line in enumerate(path.read_text().splitlines()):
            line = line.strip()
            if not line:
                continue
            m = json.loads(line)
            out.append(PromptSpec(id=f"{i:05d}", text=m["prompt"], category=m["tag"], meta=m))
        return out

    def evaluate(self, items, workdir):
        workdir = Path(workdir)
        bridge = workdir / "bridge"
        seen_meta = {}
        for it in items:
            link_or_copy(it.image_path, bridge / it.prompt.id / "samples" / f"{it.sample_idx:05d}.png")
            seen_meta.setdefault(it.prompt.id, it.prompt.meta)
        for pid, meta in seen_meta.items():
            (bridge / pid / "metadata.jsonl").write_text(json.dumps(meta))

        results = workdir / "results.jsonl"
        models = env_path("GENEVAL_MODELS", GENEVAL_ROOT / "models")
        mm = os.environ.get("GENEVAL_MM_CONFIG", str(GENEVAL_ROOT / _DEFAULT_MM))
        run([sys.executable, GENEVAL_ROOT / "evaluation" / "evaluate_images.py", bridge,
             "--outfile", results, "--model-path", models, "--model-config", mm])

        lut = {(it.prompt.id, it.sample_idx): it.item_id for it in items}
        metrics = {}
        for line in results.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            fn = Path(d["filename"])
            try:
                idx, pid = int(fn.stem), fn.parent.parent.name
            except ValueError:
                continue
            iid = lut.get((pid, idx))
            if iid:
                metrics[iid] = {"correct": int(bool(d.get("correct"))), "tag": d.get("tag")}
        return metrics

    def summarize(self, metrics, prompts):
        by_tag = {}
        for m in metrics.values():
            t = m.get("tag")
            slot = by_tag.setdefault(t, [0, 0])
            slot[0] += int(m.get("correct", 0))
            slot[1] += 1
        cats = {t: {"correct": (c[0] / c[1] if c[1] else 0.0), "n": c[1]}
                for t, c in by_tag.items()}
        macro = sum(v["correct"] for v in cats.values()) / len(cats) if cats else 0.0
        n = sum(c[1] for c in by_tag.values())
        micro = sum(int(m.get("correct", 0)) for m in metrics.values()) / n if n else 0.0
        return {"by_category": cats, "overall": {"correct": macro, "img": micro}, "n": n}
