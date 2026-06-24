"""Append-only event log: important lifecycle events, distinct from verbose logs.

Records only "what happened on the cluster" (job alloc/cancel/die, shard start/done/fail,
disk-low, campaign milestones, alerts). Single-writer (orchestrator) by default; small
JSON lines appended with O_APPEND are atomic on POSIX so occasional worker writes are safe.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from .layout import CampaignPaths


def _now_iso() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


class EventLog:
    def __init__(self, paths: CampaignPaths) -> None:
        self.paths = paths
        paths.root.mkdir(parents=True, exist_ok=True)

    def emit(self, kind: str, level: str = "INFO", **fields) -> dict:
        ev = {"ts": _now_iso(), "level": level, "kind": kind, **fields}
        with open(self.paths.events_jsonl, "a") as f:
            f.write(json.dumps(ev, ensure_ascii=False) + "\n")
        with open(self.paths.events_log, "a") as f:
            f.write(self._human(ev) + "\n")
        return ev

    @staticmethod
    def _human(ev: dict) -> str:
        base = f"{ev['ts']} [{ev['level']:>8}] {ev['kind']}"
        extra = " ".join(f"{k}={v}" for k, v in ev.items()
                         if k not in {"ts", "level", "kind"})
        return f"{base} {extra}".rstrip()

    def read(self) -> list[dict]:
        p = self.paths.events_jsonl
        if not p.is_file():
            return []
        out = []
        for line in p.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                pass
        return out
