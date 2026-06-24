"""Campaign -> work items. Status DERIVED from the filesystem (no central DB).

``expand`` produces every expected sample; ``scan_status`` recomputes done/pending by
reading the output tree, so a rerun continues exactly where it stopped (idempotent).
"""

from __future__ import annotations

import json
import random
from collections import defaultdict
from dataclasses import dataclass
from typing import Callable

from ..config import CampaignSpec, method_config_path
from .layout import CampaignPaths, ItemRef, eval_done, gen_done, gen_failed

PromptProvider = Callable[[str], list]  # bench name -> list[PromptSpec]


def _subsample(prompts: list, n, seed: int) -> list:
    """Stratified-by-category subset of ``n`` prompts (proportional per category).

    Covers all benches: geneval (per tag), oneig (per category), dpg (single
    category → plain random). Deterministic for a given seed.
    """
    if not n or n >= len(prompts):
        return prompts
    by_cat = defaultdict(list)
    for p in prompts:
        by_cat[p.category].append(p)
    cats = sorted(by_cat)
    rng = random.Random(seed)
    base, extra = divmod(n, len(cats))
    chosen = []
    for i, c in enumerate(cats):
        k = base + (1 if i < extra else 0)
        items = list(by_cat[c])
        rng.shuffle(items)
        chosen.extend(items[:k])
    chosen.sort(key=lambda p: (p.category, p.id))
    return chosen


@dataclass
class ManifestItem:
    ref: ItemRef
    text: str

    def to_json(self) -> dict:
        return {"item_id": self.ref.item_id, **self.ref.__dict__, "text": self.text}


def expand(spec: CampaignSpec, prompts_for: PromptProvider,
           samples_default: Callable[[str], int] | None = None) -> list:
    """Cross product models × configs × benchmarks × prompts × samples.

    A (model, config) pair with no resolvable method config is skipped (e.g. an
    sd35-only method when iterating cosmos2).
    """
    items: list = []
    prompt_cache: dict = {}
    for model in spec.models:
        for config in spec.configs:
            if method_config_path(model, config, spec.repo_root) is None:
                continue
            for bench in spec.benchmarks:
                n = spec.samples_per_prompt.get(bench)
                if n is None and samples_default is not None:
                    n = samples_default(bench)
                n = n or 1
                if bench not in prompt_cache:
                    prompt_cache[bench] = _subsample(
                        prompts_for(bench), spec.limit.get(bench), spec.sample_seed)
                for p in prompt_cache[bench]:
                    for s in range(n):
                        items.append(ManifestItem(
                            ItemRef(model, config, bench, p.category, p.id, s), p.text))
    return items


def write_manifest(paths: CampaignPaths, items: list) -> None:
    paths.root.mkdir(parents=True, exist_ok=True)
    with open(paths.manifest, "w") as f:
        for it in items:
            f.write(json.dumps(it.to_json(), ensure_ascii=False) + "\n")


def load_manifest(paths: CampaignPaths) -> list:
    out: list = []
    if not paths.manifest.is_file():
        return out
    for line in paths.manifest.read_text().splitlines():
        line = line.strip()
        if line:
            out.append(ItemRef.parse(json.loads(line)["item_id"]))
    return out


@dataclass
class Status:
    total: int
    gen_done: int
    eval_done: int
    pending_gen: list
    pending_eval: list
    failed: int = 0


def scan_status(paths: CampaignPaths, refs: list) -> Status:
    from .layout import read_json
    pend_gen, pend_eval = [], []
    g = e = failed = 0
    for ref in refs:
        if gen_done(paths, ref):
            g += 1
            if eval_done(paths, ref):
                e += 1
            else:
                pend_eval.append(ref)
        elif gen_failed(paths, ref):
            failed += 1
        else:
            pend_gen.append(ref)
    return Status(len(refs), g, e, pend_gen, pend_eval, failed)
