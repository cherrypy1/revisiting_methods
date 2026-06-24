"""A Shard = one worker-executable unit: (model, config, bench, category?, phase) + item ids.

Serialised to a JSON file; the worker is launched as ``cfgbench worker --shard <file>``.
Stateless per-shard execution (the loaded model is reused only within a shard).
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path


@dataclass
class Shard:
    model: str
    config: str
    bench: str
    phase: str            # "gen" | "eval"
    item_ids: list
    campaign_out: str
    category: str | None = None
    seed: int = 0
    verbose: bool = False

    @property
    def name(self) -> str:
        raw = f"{self.model}_{self.config}_{self.bench}_{self.category or 'all'}_{self.phase}"
        return re.sub(r"[^A-Za-z0-9_.-]", "_", raw)

    def to_json(self) -> dict:
        return asdict(self)

    def save(self, path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_json(), ensure_ascii=False, indent=2))
        return path

    @staticmethod
    def load(path) -> "Shard":
        return Shard(**json.loads(Path(path).read_text()))
