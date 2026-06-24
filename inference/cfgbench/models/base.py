"""ModelAdapter contract. Concrete adapters (DiffusersModelAdapter, …) land in P1.

The method (CFG/SEG/PAG/…) is NOT a separate abstraction: it is encoded in the per-model
config and applied inside ``ModelAdapter.load``. Core stays oblivious to both.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class ModelHandle(ABC):
    """A loaded model + method, ready to generate. One per shard."""

    settings: Any = None  # resolved generation params, copied into each sidecar

    @abstractmethod
    def generate(self, prompt: str, *, seed: int, n: int = 1, **override) -> list:
        """Return ``n`` PIL images for one prompt at the given seed."""

    def close(self) -> None:
        """Free VRAM / handles. Default no-op."""


class ModelAdapter(ABC):
    name: str

    @abstractmethod
    def load(self, config: dict) -> ModelHandle:
        """Load weights + apply the guidance method described by ``config``."""
