"""One-shot installer for the cfgbench benchmarking stack on the cHARISMa cluster.

Reproduces the working environment (pinned to the versions this project was validated
against) + the four external benchmark repos + the GenEval detector weights, and writes
the small runtime shims. Idempotent: re-running skips what's already present.

Usage (inside an activated venv, on the login node):
    module load gnu14/14.1
    python -m venv ~/.venv && source ~/.venv/bin/activate
    python scripts/install.py                 # everything
    python scripts/install.py --skip-deps     # only repos/weights/shims
    python scripts/install.py --only verify

Things it CANNOT do for you (printed as MANUAL steps at the end): HuggingFace login +
SD3.5/Cosmos license acceptance, the bitsandbytes 0.43.2 wheel (Cosmos only), and the
modelscope-offline patch for DPG eval.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import sysconfig
import traceback
import urllib.request
from pathlib import Path

HOME = Path.home()
REPO_ROOT = Path(__file__).resolve().parent.parent

# --- pinned versions (from the validated venv) ---
DIFFUSERS_COMMIT = "fa468c5d5"
TORCH_INDEX = "https://download.pytorch.org/whl/cu121"
PIP_TORCH = ["torch==2.5.1", "torchvision==0.20.1", "torchaudio==2.5.1"]
PIP_MIM = ["mmengine==0.10.7", "mmcv==2.1.0", "mmdet==3.3.0"]   # via openmim (CUDA-matched)
PIP_CORE = [
    f"diffusers @ git+https://github.com/huggingface/diffusers.git@{DIFFUSERS_COMMIT}",
    "transformers==4.57.3", "accelerate==1.12.0", "peft==0.18.1",
    "safetensors==0.7.0", "huggingface_hub==0.36.2", "tokenizers==0.22.2",
    "sentencepiece==0.2.1", "open_clip_torch==3.2.0", "timm==1.0.24",
    "einops==0.8.1", "ftfy==6.3.1", "modelscope==1.35.3",
    "pandas==2.3.3", "PyYAML==6.0.3", "pillow==12.0.0", "scipy==1.15.3",
    "opencv-python-headless==4.10.0.84",
]
PIP_RUNTIME_RESTORE = [
    # Some benchmark repos pin older stack pieces. Restore the generation/runtime
    # versions after those requirements so FLUX.2 imports Qwen3ForCausalLM.
    f"diffusers @ git+https://github.com/huggingface/diffusers.git@{DIFFUSERS_COMMIT}",
    "transformers==4.57.3", "accelerate==1.12.0", "huggingface_hub==0.36.2",
    "tokenizers==0.22.2", "requests>=2.32.2", "tqdm>=4.66.3",
    "urllib3>=2.2.2", "setuptools>=80.0.0",
]

# --- external benchmark repos (cfgbench env defaults point here) ---
REPOS = {
    "geneval-bench": ("https://github.com/djghosh13/geneval.git", HOME / "geneval-bench"),
    "OneIG-Benchmark": ("https://github.com/OneIG-Bench/OneIG-Benchmark.git", HOME / "OneIG-Benchmark"),
    "ELLA": ("https://github.com/TencentQQGYLab/ELLA.git", HOME / "ELLA"),
}
MMDET_REPO = "https://github.com/open-mmlab/mmdetection.git"   # for the Mask2Former py-config
M2F_URL = ("https://download.openmmlab.com/mmdetection/v2.0/mask2former/"
           "mask2former_swin-s-p4-w7-224_lsj_8x2_50e_coco/"
           "mask2former_swin-s-p4-w7-224_lsj_8x2_50e_coco_20220504_001756-743b7d99.pth")
M2F_NAME = "mask2former_swin-s-p4-w7-224_lsj_8x2_50e_coco.pth"

SOUNDFILE_STUB = '''\
# Added by cfgbench install.py: ELLA's DPG eval imports `datasets`, which imports
# `soundfile`; the cluster lacks libsndfile. Provide a no-op stub so import succeeds.
import sys, types
if "soundfile" not in sys.modules:
    _sf = types.ModuleType("soundfile")
    _sf.__libsndfile_version__ = "1.0.0"
    _sf.__version__ = "0.0.0"
    def _na(*a, **k):
        raise RuntimeError("soundfile stub (cfgbench): audio I/O unsupported on this node")
    _sf.read = _sf.write = _sf.info = _na
    sys.modules["soundfile"] = _sf
'''


def run(cmd, **kw):
    print("$ " + " ".join(map(str, cmd)), flush=True)
    subprocess.run([str(c) for c in cmd], check=True, **kw)


def pip(*args):
    run([sys.executable, "-m", "pip", "install", *args])


def step_deps():
    print("\n== Python deps ==")
    if sys.prefix == sys.base_prefix:
        print("!! Not in a virtualenv — activate one first (python -m venv ~/.venv && source ...).")
        sys.exit(1)
    pip("--upgrade", "pip", "setuptools", "wheel")
    pip("--index-url", TORCH_INDEX, *PIP_TORCH)
    pip("openmim")
    run([sys.executable, "-m", "mim", "install", *PIP_MIM])
    pip(*PIP_CORE)
    print("NOTE bitsandbytes (Cosmos only) is NOT installed here — see MANUAL steps.")


def _clone(url, dest, depth=None):
    if dest.exists():
        print(f"  {dest} exists — skip")
        return
    cmd = ["git", "clone"]
    if depth:
        cmd += ["--depth", str(depth)]
    run([*cmd, url, str(dest)])


def step_repos():
    print("\n== External benchmark repos ==")
    for name, (url, dest) in REPOS.items():
        _clone(url, dest)
    # mmdetection (config source) under geneval-bench
    _clone(MMDET_REPO, HOME / "geneval-bench" / "mmdetection", depth=1)
    # extra requirements shipped by the bench repos
    installed_extra_requirements = False
    for req in [HOME / "OneIG-Benchmark" / "requirements.txt",
                HOME / "ELLA" / "dpg_bench" / "requirements.txt"]:
        if req.is_file():
            print(f"  pip -r {req}")
            pip("-r", str(req))
            installed_extra_requirements = True
    if installed_extra_requirements:
        print("  restoring cfgbench runtime pins after benchmark requirements")
        pip("--upgrade", *PIP_RUNTIME_RESTORE)


def step_weights():
    print("\n== GenEval detector weights ==")
    models = HOME / "geneval-bench" / "models"
    models.mkdir(parents=True, exist_ok=True)
    ckpt = models / M2F_NAME
    if ckpt.is_file():
        print(f"  {ckpt} exists — skip")
        return
    print(f"  downloading Mask2Former → {ckpt}")
    urllib.request.urlretrieve(M2F_URL, ckpt)


def step_shims():
    print("\n== Runtime shims ==")
    site = Path(sysconfig.get_paths()["purelib"])
    stub = site / "sitecustomize.py"
    if stub.is_file() and "cfgbench install.py" in stub.read_text():
        print(f"  soundfile stub already in {stub} — skip")
    else:
        with stub.open("a") as f:
            f.write("\n" + SOUNDFILE_STUB)
        print(f"  wrote soundfile stub → {stub}")


def step_verify():
    print("\n== Verify ==")
    ok = True
    for mod in ("torch", "diffusers", "transformers", "mmdet", "accelerate"):
        try:
            m = __import__(mod)
            print(f"  ✓ {mod} {getattr(m, '__version__', '?')}")
        except Exception as e:
            print(f"  ✗ {mod}: {e}"); ok = False
    try:
        from transformers import Qwen2TokenizerFast, Qwen3ForCausalLM  # noqa
        print("  ✓ transformers FLUX.2 classes")
    except Exception as e:
        print(f"  ✗ transformers FLUX.2 classes: {e}"); ok = False
    try:
        from diffusers.loaders import Flux2LoraLoaderMixin  # noqa
        from diffusers.models import AutoencoderKLFlux2, Flux2Transformer2DModel  # noqa
        print("  ✓ diffusers FLUX.2 classes")
    except Exception as e:
        tb = traceback.format_exc()
        if "bitsandbytes" in tb and "0 active drivers" in tb:
            print("  ! diffusers FLUX.2 class import blocked by bitsandbytes on this node")
            print("    For Flux/SD runs uninstall bitsandbytes, or verify from a GPU node.")
        else:
            print(f"  ✗ diffusers FLUX.2 classes: {e}"); ok = False
    checks = {
        "geneval prompts": HOME / "geneval-bench" / "prompts" / "evaluation_metadata.jsonl",
        "geneval detector": HOME / "geneval-bench" / "models" / M2F_NAME,
        "mmdet config": HOME / "geneval-bench" / "mmdetection" / "configs" / "mask2former",
        "OneIG csv": HOME / "OneIG-Benchmark" / "OneIG-Bench.csv",
        "DPG prompts": HOME / "ELLA" / "dpg_bench" / "prompts",
    }
    for label, p in checks.items():
        status = "✓" if p.exists() else "✗"
        if not p.exists():
            ok = False
        print(f"  {status} {label}: {p}")
    try:
        sys.path.insert(0, str(REPO_ROOT))
        import cfgbench.benchmarks.geneval  # noqa
        import pipelines.sd35.cfg  # noqa: factory import (no model load)
        print("  ✓ cfgbench + pipeline factory import")
    except Exception as e:
        print(f"  ✗ cfgbench import: {e}"); ok = False
    print("  RESULT:", "ready" if ok else "INCOMPLETE — see messages + MANUAL steps")
    return ok


def manual_notes():
    print("\n" + "=" * 64)
    print("MANUAL steps (cannot be automated):")
    print("=" * 64)
    print(r"""
1. HuggingFace access (generation weights):
     huggingface-cli login
   Accept licenses (gated):
     https://huggingface.co/stabilityai/stable-diffusion-3.5-medium
     https://huggingface.co/nvidia/Cosmos-Predict2-2B-Text2Image
   Weights download on first run into ~/.cache/huggingface.

2. bitsandbytes 0.43.2 — ONLY needed for Cosmos2 (T5 in 4-bit nf4). The manylinux
   wheel is rejected by the cluster's default glibc, so grab + retag it:
     pip download bitsandbytes==0.43.2 --no-deps -d /tmp/bnb
     # rename the *manylinux* wheel to a plain linux tag, then:
     pip install /tmp/bnb/bitsandbytes-0.43.2-py3-none-linux_x86_64.whl
   At RUNTIME load new glibc: module load gnu14/14.1
   (Skip entirely if you only run SD3.5 + GenEval/OneIG/DPG.)

3. DPG eval (mPLUG) offline fix — modelscope 1.35.3 checks revision online and times
   out. Patch ELLA so mPLUG loads from the local cache:
     ~/ELLA/dpg_bench/compute_dpg_bench.py  → load mplug with a local model dir /
     revision, or pre-download via `modelscope download iic/mPLUG-...` and point to it.

4. Runtime always: `module load gnu14/14.1` (newer glibc) before launching the
   orchestrator; cfgbench sets TMPDIR itself.
""")


STEPS = {"deps": step_deps, "repos": step_repos, "weights": step_weights,
         "shims": step_shims, "verify": step_verify}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", choices=list(STEPS), help="run a single step")
    for s in STEPS:
        ap.add_argument(f"--skip-{s}", action="store_true")
    args = ap.parse_args()

    order = [args.only] if args.only else list(STEPS)
    for s in order:
        if getattr(args, f"skip_{s}", False):
            print(f"-- skip {s}")
            continue
        STEPS[s]()
    if not args.only or args.only == "verify":
        manual_notes()


if __name__ == "__main__":
    main()
