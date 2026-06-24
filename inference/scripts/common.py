"""Shared utilities for SD3.5 benchmark drivers."""

from __future__ import annotations

import importlib
import importlib.util
import inspect
import json
import os
import sys
from dataclasses import dataclass
from typing import Any, Callable


IMAGE_LEVEL_KEYS = {
    "steps",
    "num_inference_steps",
    "scale",
    "guidance_scale",
    "H",
    "W",
    "height",
    "width",
    "seed",
    "batch_size",
    "n_samples",
    "negative_prompt",
}

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FLUX2_MODEL_ID = "black-forest-labs/FLUX.2-klein-base-4B"
FLUX2_METHOD_DIR = os.path.join(PROJECT_ROOT, "pipelines", "flux2_klein_base")
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
    """Resolved generation settings after config/CLI merge."""

    steps: int = 25
    guidance_scale: float = 5.5
    height: int | None = None
    width: int | None = None
    seed: int = 228
    n_samples: int = 1
    negative_prompt: str | None = None
    extra: dict[str, Any] | None = None

    def call_kwargs(self, num_images: int | None = None) -> dict[str, Any]:
        kw: dict[str, Any] = dict(
            num_inference_steps=self.steps,
            guidance_scale=self.guidance_scale,
        )
        if self.negative_prompt is not None:
            kw["negative_prompt"] = self.negative_prompt
        if self.height is not None:
            kw["height"] = self.height
        if self.width is not None:
            kw["width"] = self.width
        if num_images is not None:
            kw["num_images_per_prompt"] = num_images
        if self.extra:
            kw.update(self.extra)
        return kw


def load_config(config_path: str) -> tuple[Any, dict[str, Any]]:
    """Load a pipeline config file.

    Returns ``(pipeline_spec, generation_params)``. ``pipeline_spec`` is either
    a callable ``(device) -> pipeline`` or a dict describing a ``diffusers``
    pipeline class + pretrained id.
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")

    ext = os.path.splitext(config_path)[1].lower()

    if ext == ".py":
        config_dir = os.path.dirname(os.path.abspath(config_path))
        if config_dir not in sys.path:
            sys.path.insert(0, config_dir)
        spec = importlib.util.spec_from_file_location("config_module", config_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        pipeline_spec = getattr(module, "pipeline", None)
        generation_params = dict(getattr(module, "generation_params", {}) or {})
    elif ext == ".json":
        with open(config_path) as f:
            payload = json.load(f)
        pipeline_spec = payload.get("pipeline")
        generation_params = dict(payload.get("generation_params", {}) or {})
    elif ext in (".yaml", ".yml"):
        import yaml

        with open(config_path) as f:
            payload = yaml.safe_load(f)
        pipeline_spec = payload.get("pipeline")
        generation_params = dict(payload.get("generation_params", {}) or {})
    else:
        raise ValueError(f"Unsupported config format: {ext}")

    if pipeline_spec is None:
        raise ValueError(f"Config is missing `pipeline`: {config_path}")
    return pipeline_spec, generation_params


def _resolve_class(class_name: str):
    import diffusers

    if "." in class_name:
        module_path, cls_name = class_name.rsplit(".", 1)
        module = importlib.import_module(module_path)
        return getattr(module, cls_name)
    if hasattr(diffusers, class_name):
        return getattr(diffusers, class_name)
    raise ValueError(f"Unknown pipeline class: {class_name}")


def _load_source_class(module_name: str, path: str, class_name: str):
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Pipeline source file not found: {path}")
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return getattr(module, class_name)


def _load_flux2_base(device):
    import torch

    pipeline_class = _load_source_class(
        "_scripts_flux2_cfg",
        os.path.join(FLUX2_METHOD_DIR, "cfg.py"),
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
        f"_scripts_flux2_{method}",
        os.path.join(FLUX2_METHOD_DIR, f"{method}.py"),
        class_name,
    )
    return pipeline_class.from_pipe(base)


def load_pipeline(pipeline_spec: Any, device):
    import torch

    if callable(pipeline_spec):
        return pipeline_spec(device)

    # String form: module path like "pipelines.sd35.cfg" — import the module
    # and call its `pipeline(device)` factory. Used by yaml configs.
    if isinstance(pipeline_spec, str):
        flux_prefix = "pipelines.flux2_klein_base."
        if pipeline_spec.startswith(flux_prefix):
            return _load_flux2_pipeline(pipeline_spec.removeprefix(flux_prefix), device)

        module = importlib.import_module(pipeline_spec)
        factory = getattr(module, "pipeline", None)
        if factory is None:
            raise ValueError(f"Module {pipeline_spec!r} has no `pipeline()` factory")
        return factory(device)

    cls = _resolve_class(pipeline_spec["class"])
    params = dict(pipeline_spec.get("params", {}) or {})

    if isinstance(params.get("torch_dtype"), str):
        dtype_map = {
            "float16": torch.float16,
            "float32": torch.float32,
            "bfloat16": torch.bfloat16,
        }
        params["torch_dtype"] = dtype_map.get(params["torch_dtype"], torch.float16)

    params.setdefault("torch_dtype", torch.float16)
    model = cls.from_pretrained(pipeline_spec["pretrained"], **params)
    model = model.to(device)
    if hasattr(model, "enable_attention_slicing"):
        model.enable_attention_slicing()
    return model


def resolve_settings(
    generation_params: dict[str, Any],
    overrides: dict[str, Any] | None = None,
) -> GenSettings:
    """Merge config generation_params + CLI overrides into a GenSettings.

    Keys in :data:`IMAGE_LEVEL_KEYS` populate ``GenSettings`` fields; everything
    else is forwarded verbatim via ``GenSettings.extra`` as pipeline kwargs.
    """
    merged = dict(generation_params)
    if overrides:
        for key, value in overrides.items():
            if value is None:
                continue
            merged[key] = value

    settings = GenSettings()
    extra: dict[str, Any] = {}
    for key, value in merged.items():
        if key in {"steps", "num_inference_steps"}:
            settings.steps = int(value)
        elif key in {"scale", "guidance_scale"}:
            settings.guidance_scale = float(value)
        elif key in {"H", "height"}:
            settings.height = int(value) if value is not None else None
        elif key in {"W", "width"}:
            settings.width = int(value) if value is not None else None
        elif key == "seed":
            settings.seed = int(value)
        elif key == "n_samples":
            settings.n_samples = int(value)
        elif key == "batch_size":
            pass  # unused (we always sample 1 image at a time for reproducibility)
        elif key == "negative_prompt":
            settings.negative_prompt = value
        else:
            extra[key] = value
    settings.extra = extra or None
    return settings


def write_prompt_sidecar(image_path: str, prompt: str) -> None:
    """Write a ``.txt`` file next to ``image_path`` containing the prompt."""
    root, _ = os.path.splitext(image_path)
    with open(root + ".txt", "w") as f:
        f.write(prompt.strip() + "\n")


def make_generator(device, seed: int):
    import torch

    return torch.Generator(device=device).manual_seed(int(seed))


def call_pipeline(model, prompt: str, kwargs: dict[str, Any]):
    """Call a diffusers pipeline with robust prompt/kwargs handling.

    Some custom pipelines, including FLUX.2 Klein, put ``image`` before
    ``prompt`` in ``__call__``. Use ``prompt=`` and drop unsupported kwargs for
    strict signatures.
    """
    params = inspect.signature(model.__call__).parameters
    call_kwargs = {k: v for k, v in kwargs.items() if v is not None}
    if not any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values()):
        call_kwargs = {k: v for k, v in call_kwargs.items() if k in params}
    return model(prompt=prompt, **call_kwargs)


def pick_device():
    import torch

    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
