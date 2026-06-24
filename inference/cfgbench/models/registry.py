"""Model registry. Real adapters registered in P1 (DiffusersModelAdapter for sd35/cosmos2)."""

from __future__ import annotations

from .base import ModelAdapter

MODELS: dict[str, ModelAdapter] = {}


def register(adapter: ModelAdapter) -> None:
    MODELS[adapter.name] = adapter


def get_model(name: str) -> ModelAdapter:
    if name not in MODELS:
        raise KeyError(f"model not registered: {name} (available: {sorted(MODELS)})")
    return MODELS[name]


# --- default registrations (one generic diffusers adapter per model) ---
from .diffusers_adapter import DiffusersModelAdapter  # noqa: E402

for _name in ("sd35", "cosmos2", "flux2_klein_base"):
    register(DiffusersModelAdapter(_name))
