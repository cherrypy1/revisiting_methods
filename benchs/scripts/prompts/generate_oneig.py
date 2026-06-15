"""Generate OneIG-Benchmark images for a given guidance method config.

Output layout (OneIG-Bench expects ``{out}/{short}/{method}/{id}.webp``):
    out_dir/{short}/{method_name}/{id}.webp       + {id}.txt prompt sidecar

``--grid`` selects how many samples per prompt are tiled:
    2x2 (default, original OneIG behaviour, 4 samples/grid)
    1x1 (single-sample, lets a given budget cover 4x more unique prompts)
"""

from __future__ import annotations

import argparse
import os
import sys

import pandas as pd
import torch
from PIL import Image

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
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


CATEGORY_TO_SHORT = {
    "Anime_Stylization": "anime",
    "Portrait": "portrait",
    "General_Object": "object",
    "Text_Rendering": "text",
    "Knowledge_Reasoning": "reasoning",
}

GRID_TO_N = {"1x1": 1, "2x2": 4}


def tile(images, grid):
    if grid == "1x1":
        return images[0]
    w, h = images[0].size
    canvas = Image.new("RGB", (2 * w, 2 * h))
    for idx, img in enumerate(images[:4]):
        canvas.paste(img, ((idx % 2) * w, (idx // 2) * h))
    return canvas


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("config")
    p.add_argument("--csv", required=True, help="OneIG-Bench CSV")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--model-name", required=True)
    p.add_argument(
        "--categories",
        nargs="+",
        default=list(CATEGORY_TO_SHORT.keys()),
        choices=list(CATEGORY_TO_SHORT.keys()),
    )
    p.add_argument("--grid", choices=list(GRID_TO_N.keys()), default="2x2")
    p.add_argument("--seed", type=int, default=228)
    p.add_argument("--limit", type=int, default=None, help="cap prompts per category")
    p.add_argument("--sample-seed", type=int, default=0,
                   help="RNG seed for prompt subset selection (independent of gen seed).")
    return p.parse_args()


def main():
    args = parse_args()
    n_samples = GRID_TO_N[args.grid]

    pipeline_spec, gen_params = load_config(args.config)
    settings = resolve_settings(gen_params, {"seed": args.seed})

    device = pick_device()
    model = load_pipeline(pipeline_spec, device)
    print(f"[oneig] loaded {args.config} grid={args.grid}")

    df = pd.read_csv(args.csv, dtype=str)
    df = df[df["category"].isin(args.categories)]

    for cat in args.categories:
        sub = df[df["category"] == cat]
        if args.limit and args.limit < len(sub):
            sub = sub.sample(n=args.limit, random_state=args.sample_seed)
        out_sub = os.path.join(args.out_dir, CATEGORY_TO_SHORT[cat], args.model_name)
        os.makedirs(out_sub, exist_ok=True)
        for _, row in sub.iterrows():
            prompt, pid = row["prompt_en"], row["id"]
            out_path = os.path.join(out_sub, f"{pid}.webp")
            if os.path.exists(out_path):
                continue

            images = []
            for s in range(n_samples):
                gen = make_generator(device, settings.seed + s)
                kwargs = settings.call_kwargs()
                kwargs["generator"] = gen
                images.append(model(prompt, **kwargs).images[0])

            tile(images, args.grid).save(out_path, format="webp", quality=95)
            print(f"[{cat}] {pid} saved")


if __name__ == "__main__":
    main()
