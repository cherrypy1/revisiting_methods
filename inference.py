from copy import deepcopy
from importlib import invalidate_caches, util
from pathlib import Path
import sys
from types import ModuleType


ROOT = Path(__file__).resolve().parent
FLUX2_METHOD_DIR = ROOT / "pipelines" / "Flux-2-4b-klein"
PARAMS_DIR = ROOT / "params" / "Flux-2-4b-klein"

PIPELINE_CLASS_NAMES = {
    "cfg": "Flux2KleinPipeline",
    "cfgpp": "Flux2KleinCFGPPPipeline",
    "cfg0s": "Flux2KleinCFG0SPipeline",
    "apg": "Flux2KleinAPGPipeline",
    "pag": "Flux2KleinPAGPipeline",
    "sag": "Flux2KleinSAGPipeline",
    "seg": "Flux2KleinSegPipeline",
    "oseg": "Flux2KleinOSEGPipeline",
    "tcfg": "Flux2KleinTCFGPipeline",
}


def normalize_params(params):
    if isinstance(params, tuple) and len(params) == 2:
        base_params, sweeps = params
    elif isinstance(params, dict):
        base_params = params.get("base_params", params.get("BASE_PARAMS"))
        sweeps = params.get("sweeps", params.get("SWEEPS"))
    else:
        raise TypeError("Params must be (base_params, sweeps) or a dict.")

    if base_params is None or sweeps is None:
        raise ValueError("Params must include base_params and sweeps.")

    return deepcopy(base_params), deepcopy(sweeps)


def _method_name(method):
    if isinstance(method, str):
        if method.endswith(".py") or "/" in method or "\\" in method:
            return Path(method).stem
        return method.rsplit(".", 1)[-1]

    if isinstance(method, ModuleType):
        return method.__name__.rsplit(".", 1)[-1]

    if callable(method):
        module_name = getattr(method, "__module__", "")
        if module_name.startswith("_flux2_params_"):
            return module_name.removeprefix("_flux2_params_")
        if module_name.startswith("params."):
            return module_name.rsplit(".", 1)[-1]

    raise TypeError("Method must be a method name, module, path, or params callable.")


def _load_source_module(module_name, path):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)

    invalidate_caches()
    spec = util.spec_from_file_location(module_name, path)
    module = util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def load_params(method):
    if isinstance(method, (tuple, dict)):
        return normalize_params(method)

    if callable(method) and not isinstance(method, ModuleType):
        module_name = getattr(method, "__module__", "")
        if not (
            module_name.startswith("_flux2_params_")
            or module_name.startswith("params.")
        ):
            return normalize_params(method())

    name = _method_name(method)
    module = _load_source_module(f"_flux2_params_{name}", PARAMS_DIR / f"{name}.py")

    if hasattr(module, "get_params"):
        return normalize_params(module.get_params())
    return normalize_params((module.BASE_PARAMS, module.SWEEPS))


def load_pipeline_class(method):
    name = _method_name(method)
    class_name = PIPELINE_CLASS_NAMES.get(name)
    if class_name is None:
        raise ValueError(f"Unknown method: {name}")

    module = _load_source_module(
        f"_flux2_klein_{name}",
        FLUX2_METHOD_DIR / f"{name}.py",
    )
    return getattr(module, class_name)


def reload_method_pipeline(pipe, method, **kwargs):
    pipeline_class = load_pipeline_class(method)
    return pipeline_class.from_pipe(pipe, **kwargs)


def reload_method(pipe, method, **kwargs):
    method_pipe = reload_method_pipeline(pipe, method, **kwargs)
    base_params, sweeps = load_params(method)
    return method_pipe, base_params, sweeps
