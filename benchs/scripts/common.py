"""Shared utilities for SD3.5 benchmark drivers."""

from __future__ import annotations

import importlib
import importlib.util
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
            negative_prompt=self.negative_prompt,
        )
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


def load_pipeline(pipeline_spec: Any, device):
    import torch

    if callable(pipeline_spec):
        return pipeline_spec(device)

    # String form: module path like "pipelines.sd35.cfg" — import the module
    # and call its `pipeline(device)` factory. Used by yaml configs.
    if isinstance(pipeline_spec, str):
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


def pick_device():
    import torch

    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
