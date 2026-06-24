"""Campaign + method config loading.

Campaign spec (configs/campaigns/<name>.yaml) declares WHAT to run. Method configs
(configs/methods/<model>/<name>.yaml, with current configs/<model>/<name>.yaml as fallback)
declare HOW — unchanged from today: ``pipeline:`` module-path + ``generation_params:``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent


@dataclass
class CampaignSpec:
    name: str
    models: list
    configs: list
    benchmarks: list
    out_root: Path
    stage: str = "run"                         # generate | validate | run
    samples_per_prompt: dict = field(default_factory=dict)
    seed: int = 0
    limit: dict = field(default_factory=dict)   # bench -> max prompts (subsampled)
    sample_seed: int = 0                          # seed for prompt subset selection
    pool: dict = field(default_factory=dict)
    verbose: bool = False
    repo_root: Path = REPO_ROOT


def load_campaign(path) -> CampaignSpec:
    import yaml

    path = Path(path)
    data = yaml.safe_load(path.read_text())
    out_root = data.get("out_root") or (REPO_ROOT / "outputs" / data["name"])
    stage = data.get("stage", "run")
    if stage not in {"generate", "validate", "run"}:
        raise ValueError(f"invalid campaign stage {stage!r}: expected generate|validate|run")
    return CampaignSpec(
        name=data["name"],
        models=list(data["models"]),
        configs=list(data["configs"]),
        benchmarks=list(data["benchmarks"]),
        out_root=Path(out_root),
        stage=stage,
        samples_per_prompt=dict(data.get("samples_per_prompt", {})),
        seed=int(data.get("seed", 0)),
        limit=dict(data.get("limit", {})),
        sample_seed=int(data.get("sample_seed", 0)),
        pool=dict(data.get("pool", {})),
        verbose=bool(data.get("verbose", False)),
    )


def method_config_path(model: str, name: str, repo_root: Path = REPO_ROOT):
    """Resolve a method config for (model, name); None if undefined.

    Prefers ``configs/methods/<model>/<name>.yaml``, falls back to the current
    ``configs/<model>/<name>.yaml`` so existing configs keep working until P5.
    """
    for rel in (Path("configs") / "methods" / model / f"{name}.yaml",
                Path("configs") / model / f"{name}.yaml"):
        p = repo_root / rel
        if p.is_file():
            return p
    return None


def load_method_params(path) -> dict:
    import yaml

    data = yaml.safe_load(Path(path).read_text()) or {}
    return dict(data.get("generation_params", {}) or {})
