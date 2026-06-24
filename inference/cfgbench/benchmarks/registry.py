"""Benchmark registry. Real adapters (geneval/oneig/dpg) registered in P1."""

from __future__ import annotations

from .base import BenchmarkAdapter

BENCHMARKS: dict[str, BenchmarkAdapter] = {}


def register(adapter: BenchmarkAdapter) -> None:
    BENCHMARKS[adapter.name] = adapter


def get_benchmark(name: str) -> BenchmarkAdapter:
    if name not in BENCHMARKS:
        raise KeyError(f"benchmark not registered: {name} (available: {sorted(BENCHMARKS)})")
    return BENCHMARKS[name]


def _register_defaults() -> None:
    from .dpg import DPGAdapter
    from .geneval import GenEvalAdapter
    from .oneig import OneIGAdapter

    for adapter in (GenEvalAdapter(), OneIGAdapter(), DPGAdapter()):
        register(adapter)


_register_defaults()
