"""Generic diffusers model adapter — covers sd35, cosmos2, and most diffusers models.

Driven entirely by the method config (``pipeline:`` module-path + ``generation_params:``).
Self-contained port of the proven logic from the legacy ``scripts/common.py`` so cfgbench
does not depend on ``scripts/`` (which is deleted in P5). All heavy imports (torch, diffusers)
are lazy so importing this module — e.g. for the registry — costs nothing.
"""

from __future__ import annotations

import importlib
import importlib.util
import inspect
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .base import ModelAdapter, ModelHandle

IMAGE_LEVEL_KEYS = {
    "steps", "num_inference_steps", "scale", "guidance_scale",
    "H", "W", "height", "width", "seed", "batch_size", "n_samples", "negative_prompt",
}

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
FLUX2_MODEL_ID = "black-forest-labs/FLUX.2-klein-base-4B"
FLUX2_METHOD_DIR = REPO_ROOT / "pipelines" / "flux2_klein_base"
FLUX2_CLASS_NAMES = {
    "apg": "Flux2KleinAPGPipeline",
    "cfg0s": "Flux2KleinCFG0SPipeline",
    "cfgpp": "Flux2KleinCFGPPPipeline",
    "oseg": "Flux2KleinOSEGPipeline",
    "pag": "Flux2KleinPAGPipeline",
    "sag": "Flux2KleinSAGPipeline",
    "seg": "Flux2KleinSegPipeline",
    "tcfg": "Flux2KleinTCFGPipeline",
}


@dataclass
class GenSettings:
    steps: int = 25
    guidance_scale: float = 5.5
    height: int | None = None
    width: int | None = None
    seed: int = 228
    n_samples: int = 1
    negative_prompt: str | None = None
    extra: dict | None = None

    def call_kwargs(self) -> dict:
        kw: dict = dict(num_inference_steps=self.steps,
                        guidance_scale=self.guidance_scale)
        if self.negative_prompt is not None:
            kw["negative_prompt"] = self.negative_prompt
        if self.height is not None:
            kw["height"] = self.height
        if self.width is not None:
            kw["width"] = self.width
        if self.extra:
            kw.update(self.extra)
        return kw

    def as_params(self) -> dict:
        """Serialisable params for the sidecar (drops None / runtime objects)."""
        d = {k: v for k, v in self.call_kwargs().items() if v is not None}
        d.pop("generator", None)
        return d


def load_config(config_path: str):
    """Return (pipeline_spec, generation_params) from a .yaml/.json/.py config."""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    ext = os.path.splitext(config_path)[1].lower()

    if ext in (".yaml", ".yml"):
        import yaml
        payload = yaml.safe_load(Path(config_path).read_text())
        spec = payload.get("pipeline")
        params = dict(payload.get("generation_params", {}) or {})
    elif ext == ".json":
        payload = json.loads(Path(config_path).read_text())
        spec = payload.get("pipeline")
        params = dict(payload.get("generation_params", {}) or {})
    elif ext == ".py":
        cdir = os.path.dirname(os.path.abspath(config_path))
        import sys
        if cdir not in sys.path:
            sys.path.insert(0, cdir)
        m = importlib.util.spec_from_file_location("config_module", config_path)
        mod = importlib.util.module_from_spec(m)
        m.loader.exec_module(mod)
        spec = getattr(mod, "pipeline", None)
        params = dict(getattr(mod, "generation_params", {}) or {})
    else:
        raise ValueError(f"Unsupported config format: {ext}")

    if spec is None:
        raise ValueError(f"Config is missing `pipeline`: {config_path}")
    return spec, params


def _load_source_class(module_name: str, path: Path, class_name: str):
    if not path.is_file():
        raise FileNotFoundError(f"Pipeline source file not found: {path}")
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return getattr(module, class_name)


def _load_flux2_base(device):
    import torch

    pipeline_class = _load_source_class(
        "_cfgbench_flux2_cfg",
        FLUX2_METHOD_DIR / "cfg.py",
        "Flux2KleinPipeline",
    )

    kwargs = {"torch_dtype": torch.float16}
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
    if token:
        kwargs["token"] = token

    device_type = getattr(device, "type", str(device))
    if device_type == "cuda":
        kwargs["device_map"] = "cuda"

    pipe = pipeline_class.from_pretrained(
        os.environ.get("FLUX2_MODEL_ID", FLUX2_MODEL_ID),
        **kwargs,
    )
    if "device_map" not in kwargs:
        pipe = pipe.to(device)
    return pipe


def _load_flux2_pipeline(method: str, device):
    base = _load_flux2_base(device)
    if method in {"cfg", "no_cfg"}:
        return base

    class_name = FLUX2_CLASS_NAMES.get(method)
    if class_name is None:
        raise ValueError(f"Unknown Flux2 method: {method}")

    pipeline_class = _load_source_class(
        f"_cfgbench_flux2_{method}",
        FLUX2_METHOD_DIR / f"{method}.py",
        class_name,
    )
    return pipeline_class.from_pipe(base)


def load_pipeline(pipeline_spec: Any, device):
    """Instantiate the pipeline. ``pipeline_spec`` is a callable, a module-path
    string with a ``pipeline(device)`` factory, or a dict {class, pretrained, params}."""
    import torch

    if callable(pipeline_spec):
        return pipeline_spec(device)

    if isinstance(pipeline_spec, str):
        flux_prefix = "pipelines.flux2_klein_base."
        if pipeline_spec.startswith(flux_prefix):
            return _load_flux2_pipeline(pipeline_spec.removeprefix(flux_prefix), device)

        module = importlib.import_module(pipeline_spec)
        factory = getattr(module, "pipeline", None)
        if factory is None:
            raise ValueError(f"Module {pipeline_spec!r} has no `pipeline()` factory")
        return factory(device)

    # dict form: diffusers class + pretrained id
    import diffusers
    cls_name = pipeline_spec["class"]
    if "." in cls_name:
        modpath, name = cls_name.rsplit(".", 1)
        cls = getattr(importlib.import_module(modpath), name)
    else:
        cls = getattr(diffusers, cls_name)
    params = dict(pipeline_spec.get("params", {}) or {})
    if isinstance(params.get("torch_dtype"), str):
        params["torch_dtype"] = {"float16": torch.float16, "float32": torch.float32,
                                 "bfloat16": torch.bfloat16}.get(params["torch_dtype"],
                                                                 torch.float16)
    params.setdefault("torch_dtype", torch.float16)
    model = cls.from_pretrained(pipeline_spec["pretrained"], **params).to(device)
    if hasattr(model, "enable_attention_slicing"):
        model.enable_attention_slicing()
    return model


def resolve_settings(generation_params: dict, overrides: dict | None = None) -> GenSettings:
    merged = dict(generation_params)
    for k, v in (overrides or {}).items():
        if v is not None:
            merged[k] = v
    s = GenSettings()
    extra: dict = {}
    for k, v in merged.items():
        if k in {"steps", "num_inference_steps"}:
            s.steps = int(v)
        elif k in {"scale", "guidance_scale"}:
            s.guidance_scale = float(v)
        elif k in {"H", "height"}:
            s.height = int(v) if v is not None else None
        elif k in {"W", "width"}:
            s.width = int(v) if v is not None else None
        elif k == "seed":
            s.seed = int(v)
        elif k == "n_samples":
            s.n_samples = int(v)
        elif k == "batch_size":
            pass
        elif k == "negative_prompt":
            s.negative_prompt = v
        else:
            extra[k] = v
    s.extra = extra or None
    return s


def _make_generator(device, seed: int):
    import torch
    return torch.Generator(device=device).manual_seed(int(seed))


def _pick_device():
    import torch
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _pipeline_call(pipe, prompt: str, kwargs: dict):
    """Call heterogeneous diffusers pipelines with prompt as a keyword.

    FLUX.2 Klein has ``image`` before ``prompt`` in ``__call__``; positional
    prompt calls therefore pass text as an image. Some pipelines also reject
    kwargs that other pipelines accept, so filter by signature unless the
    pipeline explicitly accepts ``**kwargs``.
    """
    params = inspect.signature(pipe.__call__).parameters
    call_kwargs = {k: v for k, v in kwargs.items() if v is not None}
    if not any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values()):
        call_kwargs = {k: v for k, v in call_kwargs.items() if k in params}
    return pipe(prompt=prompt, **call_kwargs)


class _DiffusersHandle(ModelHandle):
    def __init__(self, pipe, settings: GenSettings, device):
        self.pipe = pipe
        self.settings = settings
        self.device = device

    def params(self) -> dict:
        return self.settings.as_params()

    def generate(self, prompt: str, *, seed: int, n: int = 1, **override):
        base = self.settings.call_kwargs()
        base.update({k: v for k, v in override.items() if v is not None})
        imgs = []
        for i in range(n):
            kw = dict(base)
            kw["num_images_per_prompt"] = 1
            kw["generator"] = _make_generator(self.device, seed + i)
            imgs.append(_pipeline_call(self.pipe, prompt, kw).images[0])
        return imgs

    def close(self):
        try:
            import gc
            import torch
            del self.pipe
            gc.collect()
            torch.cuda.empty_cache()
        except Exception:
            pass


class DiffusersModelAdapter(ModelAdapter):
    """One adapter for any diffusers checkpoint; differences live in pipelines/ + configs/."""

    def __init__(self, name: str):
        self.name = name

    def load(self, config) -> _DiffusersHandle:
        path = config["__path__"] if isinstance(config, dict) else config
        spec, gen_params = load_config(str(path))
        settings = resolve_settings(gen_params)
        device = _pick_device()
        pipe = load_pipeline(spec, device)
        return _DiffusersHandle(pipe, settings, device)
