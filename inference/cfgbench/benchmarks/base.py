"""BenchmarkAdapter contract + shared data types + default aggregation.

Core owns generation, output paths and sidecars. An adapter only: supplies prompts,
evaluates generated items into per-item metrics, and (optionally) overrides summarize.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from statistics import mean


@dataclass(frozen=True)
class PromptSpec:
    id: str
    text: str
    category: str
    meta: dict = field(default_factory=dict)


@dataclass(frozen=True)
class GenItem:
    """One generated sample handed to ``evaluate``."""

    item_id: str
    prompt: PromptSpec
    image_path: Path
    sidecar_path: Path
    sample_idx: int


def _numeric(d: dict) -> dict:
    return {k: v for k, v in d.items()
            if isinstance(v, (int, float)) and not isinstance(v, bool)}


def _mean_rows(rows: list[dict]) -> dict:
    keys: set = set().union(*[r.keys() for r in rows]) if rows else set()
    out = {k: mean([r[k] for r in rows if k in r])
           for k in keys if any(k in r for r in rows)}
    out["n"] = len(rows)
    return out


def default_summarize(metrics: dict[str, dict], prompts) -> dict:
    """Mean of each numeric metric, grouped by prompt category, plus overall."""
    pid_cat = {p.id: p.category for p in prompts}
    by_cat: dict[str, list[dict]] = {}
    for item_id, m in metrics.items():
        parts = item_id.split("/")
        prompt_id = parts[-2] if len(parts) >= 2 else item_id
        cat = pid_cat.get(prompt_id, "all")
        by_cat.setdefault(cat, []).append(_numeric(m))
    all_rows = [r for rows in by_cat.values() for r in rows]
    return {
        "by_category": {cat: _mean_rows(rows) for cat, rows in by_cat.items()},
        "overall": _mean_rows(all_rows),
        "n": len(all_rows),
    }


class BenchmarkAdapter(ABC):
    name: str
    eval_needs_gpu: bool = True
    default_samples_per_prompt: int = 1

    @abstractmethod
    def prompts(self) -> list[PromptSpec]:
        ...

    @abstractmethod
    def evaluate(self, items: list[GenItem], workdir: Path) -> dict[str, dict]:
        """Return ``item_id -> metrics``. ``workdir`` is private scratch for bridging."""

    def summarize(self, metrics: dict[str, dict], prompts) -> dict:
        return default_summarize(metrics, prompts)
