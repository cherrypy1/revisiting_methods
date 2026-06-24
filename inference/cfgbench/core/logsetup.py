"""Two-level logging (basic INFO / verbose DEBUG) + rolling throughput meter.

- basic: run/shard start-end, allocation events, summaries.
- verbose: per-image start + rolling images/s over a trailing window.
"""

from __future__ import annotations

import logging
import sys
import time
from collections import deque
from pathlib import Path

_FMT = "%(asctime)s [%(levelname)s] %(message)s"
_DATEFMT = "%H:%M:%S"


def configure(verbose: bool = False, logfile=None, name: str = "cfgbench") -> logging.Logger:
    logger = logging.getLogger(name)
    logger.handlers.clear()
    logger.setLevel(logging.DEBUG if verbose else logging.INFO)
    fmt = logging.Formatter(_FMT, _DATEFMT)

    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(sh)

    if logfile:
        Path(logfile).parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(logfile)
        fh.setFormatter(fmt)
        logger.addHandler(fh)

    logger.propagate = False
    return logger


class ThroughputMeter:
    """Rolling images/sec over a trailing time window."""

    def __init__(self, window_s: float = 60.0) -> None:
        self.window_s = window_s
        self._ts: deque = deque()
        self.total = 0

    def tick(self, n: int = 1) -> None:
        now = time.monotonic()
        for _ in range(n):
            self._ts.append(now)
        self.total += n
        self._trim(now)

    def _trim(self, now: float) -> None:
        cut = now - self.window_s
        while self._ts and self._ts[0] < cut:
            self._ts.popleft()

    def rate(self) -> float:
        now = time.monotonic()
        self._trim(now)
        if len(self._ts) < 2:
            return 0.0
        span = now - self._ts[0]
        return len(self._ts) / span if span > 0 else 0.0

    def eta_s(self, remaining: int):
        r = self.rate()
        return remaining / r if r > 0 else None
