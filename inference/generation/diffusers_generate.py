"""GenEval-layout generator for SD3.5 guidance methods.

Writes ``{outdir}/{NNNNN}/samples/{MMMMM}.png`` + ``.txt`` prompt sidecar
and ``{outdir}/{NNNNN}/metadata.jsonl`` per prompt, matching the layout
consumed by ``evaluation/evaluate_images.py``.
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import random
import sys

import torch
from pytorch_lightning import seed_everything

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from scripts.common import (  # noqa: E402
    load_config,
    load_pipeline,
    make_generator,
    pick_device,
    resolve_settings,
    write_prompt_sidecar,
)


torch.set_grad_enabled(False)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("metadata_file", help="JSONL with one {prompt,...} per line")
    p.add_argument("--config", required=True, help="Path to pipeline config (.py/.json/.yaml)")
    p.add_argument("--outdir", default="outputs")
    p.add_argument("--n_samples", type=int, default=None)
    p.add_argument("--steps", type=int, default=None)
    p.add_argument("--scale", type=float, default=None, dest="guidance_scale")
    p.add_argument("--H", type=int, default=None, dest="height")
    p.add_argument("--W", type=int, default=None, dest="width")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--negative_prompt", default=None)
    p.add_argument("--skip_grid", action="store_true", help="(kept for CLI compat; no grid is written either way)")
    p.add_argument("--limit", type=int, default=None,
                   help="Total prompt budget; sampled stratified-random across tags.")
    p.add_argument("--sample-seed", type=int, default=0,
                   help="RNG seed for prompt subset selection (independent of gen seed).")
    return p.parse_args()


def stratified_sample(metadatas, limit, seed):
    by_tag = collections.defaultdict(list)
    for m in metadatas:
        by_tag[m["tag"]].append(m)
    tags = sorted(by_tag.keys())
    rng = random.Random(seed)
    base, extra = divmod(limit, len(tags))
    chosen = []
    for i, t in enumerate(tags):
        n = base + (1 if i < extra else 0)
        items = list(by_tag[t])
        rng.shuffle(items)
        chosen.extend(items[:n])
    rng.shuffle(chosen)
    return chosen


def main():
    args = parse_args()

    pipeline_spec, gen_params = load_config(args.config)
    overrides = {
        k: v for k, v in vars(args).items()
        if k in {"steps", "guidance_scale", "height", "width", "seed", "n_samples", "negative_prompt"}
    }
    settings = resolve_settings(gen_params, overrides)

    with open(args.metadata_file) as fp:
        metadatas = [json.loads(line) for line in fp]
    if args.limit and args.limit < len(metadatas):
        metadatas = stratified_sample(metadatas, args.limit, args.sample_seed)

    device = pick_device()
    model = load_pipeline(pipeline_spec, device)
    print(f"[geneval] loaded {args.config}")

    for idx, metadata in enumerate(metadatas):
        seed_everything(settings.seed)
        outpath = os.path.join(args.outdir, f"{idx:05d}")
        sample_path = os.path.join(outpath, "samples")
        os.makedirs(sample_path, exist_ok=True)

        prompt = metadata["prompt"]
        print(f"[geneval] ({idx+1}/{len(metadatas)}) {prompt!r}")

        with open(os.path.join(outpath, "metadata.jsonl"), "w") as fp:
            json.dump(metadata, fp)

        kwargs = settings.call_kwargs(num_images=settings.n_samples)
        kwargs["generator"] = make_generator(device, settings.seed)
        images = model(prompt, **kwargs).images

        for j, img in enumerate(images):
            img_path = os.path.join(sample_path, f"{j:05d}.png")
            img.save(img_path)
            write_prompt_sidecar(img_path, prompt)

    print("Done.")


if __name__ == "__main__":
    main()
