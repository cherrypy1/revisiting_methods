"""Uniform output layout + atomic sidecar IO. Core-owned; oblivious to model/bench.

A campaign's results live under one root::

    <out_root>/
        manifest.jsonl  pool.json  events.jsonl  events.log  status.json  ALERT
        heartbeats/<jobid>.json
        logs/<jobid>__<shard>.log  logs/orchestrator.log
        <model>/<config>/<bench>/<category>/<prompt_id>/
            sample_<idx>.png
            sample_<idx>.json          # per-image sidecar (prompt + params + metrics)
        <model>/<config>/<bench>/<category>/_summary.json
        <model>/<config>/<bench>/_summary.json
        <model>/<config>/_summary.json

Status is DERIVED from this tree (see gen_done / eval_done) — there is no central DB.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ItemRef:
    """Identity of one generated sample. Also fully determines its output paths."""

    model: str
    config: str
    bench: str
    category: str
    prompt_id: str
    sample_idx: int

    @property
    def item_id(self) -> str:
        return (f"{self.model}/{self.config}/{self.bench}/"
                f"{self.category}/{self.prompt_id}/s{self.sample_idx}")

    @staticmethod
    def parse(item_id: str) -> "ItemRef":
        model, config, bench, category, prompt_id, s = item_id.split("/")
        return ItemRef(model, config, bench, category, prompt_id, int(s[1:]))


def write_json_atomic(path, obj) -> None:
    """Write JSON via tmp + os.replace so a partial file never looks complete."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2))
    os.replace(tmp, path)  # atomic on POSIX


def read_json(path):
    path = Path(path)
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return None


def image_ok(path) -> bool:
    """True if the image exists and is decodable. Falls back to size>0 if PIL absent."""
    path = Path(path)
    if not path.is_file() or path.stat().st_size == 0:
        return False
    try:
        from PIL import Image
    except ImportError:
        return True
    try:
        with Image.open(path) as im:
            im.verify()
        return True
    except Exception:
        return False


class CampaignPaths:
    """All filesystem locations for one campaign, derived from ``out_root``."""

    def __init__(self, out_root) -> None:
        self.root = Path(out_root)

    # ---- campaign-level files ----
    @property
    def manifest(self):
        return self.root / "manifest.jsonl"

    @property
    def pool(self):
        return self.root / "pool.json"

    @property
    def events_jsonl(self):
        return self.root / "events.jsonl"

    @property
    def events_log(self):
        return self.root / "events.log"

    @property
    def status(self):
        return self.root / "status.json"

    @property
    def alert(self):
        return self.root / "ALERT"

    @property
    def heartbeats(self):
        return self.root / "heartbeats"

    @property
    def logs(self):
        return self.root / "logs"

    def heartbeat(self, jobid):
        return self.heartbeats / f"{jobid}.json"

    def worker_log(self, jobid, shard):
        return self.logs / f"{jobid}__{shard}.log"

    def orchestrator_log(self):
        return self.logs / "orchestrator.log"

    # ---- result tree ----
    def run_dir(self, model, config):
        return self.root / model / config

    def bench_dir(self, model, config, bench):
        return self.run_dir(model, config) / bench

    def category_dir(self, model, config, bench, category):
        return self.bench_dir(model, config, bench) / category

    def item_dir(self, ref: ItemRef):
        return self.category_dir(ref.model, ref.config, ref.bench, ref.category) / ref.prompt_id

    def image(self, ref: ItemRef):
        return self.item_dir(ref) / f"sample_{ref.sample_idx}.png"

    def sidecar(self, ref: ItemRef):
        return self.item_dir(ref) / f"sample_{ref.sample_idx}.json"

    # ---- summaries ----
    def summary_category(self, model, config, bench, category):
        return self.category_dir(model, config, bench, category) / "_summary.json"

    def summary_bench(self, model, config, bench):
        return self.bench_dir(model, config, bench) / "_summary.json"

    def summary_run(self, model, config):
        return self.run_dir(model, config) / "_summary.json"

    def ensure(self) -> "CampaignPaths":
        for d in (self.root, self.heartbeats, self.logs):
            d.mkdir(parents=True, exist_ok=True)
        return self


def build_sidecar(ref: ItemRef, *, text, seed, params, gen_time_s,
                  status="ok", metrics=None) -> dict:
    return {
        "item_id": ref.item_id,
        "model": ref.model,
        "config": ref.config,
        "bench": ref.bench,
        "category": ref.category,
        "prompt_id": ref.prompt_id,
        "sample_idx": ref.sample_idx,
        "text": text,
        "seed": seed,
        "params": params or {},
        "gen_time_s": gen_time_s,
        "status": status,
        "metrics": metrics or {},
    }


def gen_done(paths: CampaignPaths, ref: ItemRef) -> bool:
    """Generation complete ⇔ sidecar present AND image decodable."""
    return bool(read_json(paths.sidecar(ref))) and image_ok(paths.image(ref))


def gen_failed(paths: CampaignPaths, ref: ItemRef) -> bool:
    """Generation failed permanently for this run."""
    side = read_json(paths.sidecar(ref))
    return bool(side) and side.get("status") == "failed"


def eval_done(paths: CampaignPaths, ref: ItemRef) -> bool:
    """Eval complete ⇔ sidecar present with a non-empty metrics block."""
    side = read_json(paths.sidecar(ref))
    return bool(side) and bool(side.get("metrics"))
