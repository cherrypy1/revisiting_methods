"""Cosmos2 hyperparameter search: load model once, sweep small grids per method.

Output layout:
  outputs/cosmos2/hp/{method}/{trial}/{idx:05d}/samples/00000.png
  outputs/cosmos2/hp/{method}/{trial}/{idx:05d}/metadata.jsonl

Compatible with `evaluation/evaluate_images.py` so each {method}/{trial}
folder can be scored independently.
"""

from __future__ import annotations

import argparse
import collections
import itertools
import json
import os
import random
import sys
import time
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.common import make_generator, pick_device, write_prompt_sidecar  # noqa: E402

# GenEval prompt set used as the stratified sampling source (external repo).
GENEVAL_ROOT = Path(os.environ.get("GENEVAL_ROOT", str(Path.home() / "geneval-bench")))
DEFAULT_PROMPTS = GENEVAL_ROOT / "prompts" / "evaluation_metadata.jsonl"
DEFAULT_OUT = PROJECT_ROOT / "outputs" / "cosmos2" / "hp"

# Per-method grids. Designed so each method generates ≤72 images
# (≤4 trials × 18 prompts).
# Cosmos-Predict2-2B: ComfyUI/NVIDIA defaults use CFG=1.0-2.0 (model trained for
# low-CFG regime, unlike SD3.5). High CFG (5+) oversaturates and amplifies
# additive PAG/SEG terms into artifacts. Grids span low + mid ranges.
# Cosmos2-2B transformer has 28 blocks; PAG/SEG layer choices probe early
# (structure), mid (semantics), late (fine detail).
GRIDS = {
    "cfg":    [{"guidance_scale": s} for s in (1.0, 1.5, 2.0, 3.0, 5.0, 7.0)],
    "no_cfg": [{"guidance_scale": 0.0}],
    "cfgpp":  [{"guidance_scale": g, "cfgpp_w": w}
               for g in (1.5, 2.5) for w in (0.5, 0.7, 1.0)],
    "cfg0s":  [
        {"guidance_scale": 1.5, "zero_steps": 0},
        {"guidance_scale": 2.0, "zero_steps": 0},
        {"guidance_scale": 2.0, "zero_steps": 2},
        {"guidance_scale": 3.0, "zero_steps": 0},
    ],
    "apg": [
        {"guidance_scale": 1.5, "apg_momentum": -0.5},
        {"guidance_scale": 2.5, "apg_momentum": -0.5},
        {"guidance_scale": 2.5, "apg_momentum": 0.0},
        {"guidance_scale": 5.0, "apg_momentum": -0.5},
    ],
    "tcfg":  [{"guidance_scale": s, "tcfg_rank": 1} for s in (1.5, 2.0, 3.0, 5.0)],
    "sag": [
        {"guidance_scale": 1.5, "sag_scale": 0.2, "sag_blur_sigma": 1.0},
        {"guidance_scale": 2.0, "sag_scale": 0.4, "sag_blur_sigma": 1.0},
        {"guidance_scale": 2.0, "sag_scale": 0.75, "sag_blur_sigma": 1.0},
    ],
    # PAG: lower base CFG + lower pag_scale + probe mid-block (paper recs).
    # arxiv 2506.10978 finds mid-layer perturb best for DiTs.
    "pag": [
        {"guidance_scale": 1.5, "pag_scale": 1.0, "pag_applied_layers": [14]},
        {"guidance_scale": 2.0, "pag_scale": 1.5, "pag_applied_layers": [14]},
        {"guidance_scale": 2.0, "pag_scale": 1.5, "pag_applied_layers": [1]},
        {"guidance_scale": 2.0, "pag_scale": 1.5, "pag_applied_layers": [12, 14, 16]},
        {"guidance_scale": 2.0, "pag_scale": 3.0, "pag_applied_layers": [14]},
    ],
    "seg": [
        {"guidance_scale": 1.5, "seg_scale": 1.0, "seg_blur_sigma": 10.0, "seg_applied_layers": [14]},
        {"guidance_scale": 2.0, "seg_scale": 1.5, "seg_blur_sigma": 10.0, "seg_applied_layers": [14]},
        {"guidance_scale": 2.0, "seg_scale": 3.0, "seg_blur_sigma": 10.0, "seg_applied_layers": [8]},
        {"guidance_scale": 3.0, "seg_scale": 3.0, "seg_blur_sigma": 10.0, "seg_applied_layers": [14]},
    ],
    "oseg": [
        {"guidance_scale": 1.5, "seg_scale": 1.0, "oseg_scale": 0.5,
         "seg_blur_sigma": 1.0e10, "seg_applied_layers": [14]},
        {"guidance_scale": 2.0, "seg_scale": 2.0, "oseg_scale": 0.5,
         "seg_blur_sigma": 1.0e10, "seg_applied_layers": [14]},
        {"guidance_scale": 2.0, "seg_scale": 2.0, "oseg_scale": 1.0,
         "seg_blur_sigma": 1.0e10, "seg_applied_layers": [8]},
    ],
}

DEFAULTS = {
    "num_inference_steps": 25,
    "height": 1024,
    "width": 1024,
    "negative_prompt": ("worst quality, low quality, blurry, jpeg artifacts, "
                        "deformed, distorted, ugly"),
}


def stratified_sample(metadatas, per_tag, seed):
    by_tag = collections.defaultdict(list)
    for m in metadatas:
        by_tag[m["tag"]].append(m)
    rng = random.Random(seed)
    chosen = []
    for tag in sorted(by_tag):
        items = list(by_tag[tag])
        rng.shuffle(items)
        chosen.extend(items[:per_tag])
    rng.shuffle(chosen)
    return chosen


def trial_name(params: dict) -> str:
    parts = []
    for k, v in sorted(params.items()):
        if isinstance(v, float):
            v = f"{v:g}"
        elif isinstance(v, list):
            v = "_".join(str(x) for x in v)
        parts.append(f"{k}={v}")
    return ".".join(parts).replace("guidance_scale", "g").replace("_scale", "s")[:120]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--methods", nargs="+", default=list(GRIDS.keys()))
    p.add_argument("--prompts", default=str(DEFAULT_PROMPTS))
    p.add_argument("--out-root", default=str(DEFAULT_OUT))
    p.add_argument("--per-tag", type=int, default=3,
                   help="prompts per geneval category; 6 cats * per_tag = total prompts/trial")
    p.add_argument("--seed", type=int, default=228)
    p.add_argument("--sample-seed", type=int, default=0)
    p.add_argument("--steps", type=int, default=DEFAULTS["num_inference_steps"])
    p.add_argument("--resolution", type=int, default=DEFAULTS["height"])
    p.add_argument("--method-config", default=str(PROJECT_ROOT / "configs" / "cosmos2" / "cfg.yaml"),
                   help="any cosmos2 config (used only to load the pipeline once)")
    return p.parse_args()


def main():
    args = parse_args()
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    with open(args.prompts) as fp:
        metadatas = [json.loads(line) for line in fp]
    chosen = stratified_sample(metadatas, args.per_tag, args.sample_seed)
    print(f"[hp] {len(chosen)} prompts per trial; methods={args.methods}")

    device = pick_device()
    # Lazy-load pipeline using the same _patch so we get bnb-quantized T5.
    from pipelines.cosmos2._patch import load_pipeline as build_pipe  # noqa: E402
    print(f"[hp] loading pipeline...")
    t0 = time.time()
    pipe = build_pipe("cfg", device)
    print(f"[hp] loaded in {time.time()-t0:.1f}s")

    for method in args.methods:
        if method not in GRIDS:
            print(f"[hp] skip unknown method {method}")
            continue
        method_root = out_root / method
        method_root.mkdir(parents=True, exist_ok=True)
        for trial in GRIDS[method]:
            tname = trial_name(trial)
            trial_root = method_root / tname
            trial_root.mkdir(parents=True, exist_ok=True)
            kwargs_template = {
                **DEFAULTS,
                "method": method,
                "num_inference_steps": args.steps,
                "height": args.resolution,
                "width": args.resolution,
                **trial,
            }
            print(f"\n[hp] === {method} :: {tname} ===")
            t_start = time.time()
            for idx, meta in enumerate(chosen):
                outpath = trial_root / f"{idx:05d}"
                sample_path = outpath / "samples"
                sample_path.mkdir(parents=True, exist_ok=True)
                img_path = sample_path / "00000.png"
                with open(outpath / "metadata.jsonl", "w") as fp:
                    json.dump(meta, fp)
                if img_path.exists():
                    continue
                gen = make_generator(device, args.seed)
                kwargs = dict(kwargs_template)
                kwargs["generator"] = gen
                with torch.no_grad():
                    image = pipe(meta["prompt"], **kwargs).images[0]
                image.save(img_path)
                write_prompt_sidecar(str(img_path), meta["prompt"])
            print(f"[hp] {method}/{tname} done in {time.time()-t_start:.1f}s")

    print("[hp] ALL DONE")


if __name__ == "__main__":
    main()
