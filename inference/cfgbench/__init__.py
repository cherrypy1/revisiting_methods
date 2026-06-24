"""cfgbench — generic, autonomous, resumable benchmark runner for guidance methods.

Core is oblivious to model/benchmark internals; only adapters (cfgbench.models,
cfgbench.benchmarks) know about diffusers / mmdet / OneIG / DPG. See PLAN.md + ADAPTERS.md.
"""

__version__ = "0.0.1"
