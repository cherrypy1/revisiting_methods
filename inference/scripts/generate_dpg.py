"""Generate DPG-Bench images for a given SD3.5 guidance method.

Output: ``{out_dir}/{id}.png`` (horizontal tile of ``pic-num`` samples, which is
the layout ``ELLA/dpg_bench/compute_dpg_bench.py`` expects). A ``{id}.txt``
prompt sidecar is written alongside every image.
"""

from __future__ import annotations

import argparse
import glob
import os
import random
import sys

from PIL import Image

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from scripts.common import (  # noqa: E402
    call_pipeline,
    load_config,
    load_pipeline,
    make_generator,
    pick_device,
    resolve_settings,
    write_prompt_sidecar,
)


def horizontal_tile(images):
    if len(images) == 1:
        return images[0]
    w, h = images[0].size
    canvas = Image.new("RGB", (w * len(images), h))
    for idx, img in enumerate(images):
        canvas.paste(img, (w * idx, 0))
    return canvas


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("config")
    p.add_argument("--prompts-dir", required=True, help="dpg_bench/prompts")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--pic-num", type=int, default=4)
    p.add_argument("--resolution", type=int, default=1024)
    p.add_argument("--seed", type=int, default=228)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--sample-seed", type=int, default=0,
                   help="RNG seed for prompt subset selection (independent of gen seed).")
    return p.parse_args()


def main():
    args = parse_args()
    pipeline_spec, gen_params = load_config(args.config)
    overrides = {
        "seed": args.seed,
        "height": args.resolution,
        "width": args.resolution,
    }
    settings = resolve_settings(gen_params, overrides)

    device = pick_device()
    model = load_pipeline(pipeline_spec, device)
    print(f"[dpg] loaded {args.config}")

    os.makedirs(args.out_dir, exist_ok=True)
    prompt_files = sorted(glob.glob(os.path.join(args.prompts_dir, "*.txt")))
    if args.limit and args.limit < len(prompt_files):
        rng = random.Random(args.sample_seed)
        prompt_files = sorted(rng.sample(prompt_files, args.limit))

    for path in prompt_files:
        idx = os.path.splitext(os.path.basename(path))[0]
        out_path = os.path.join(args.out_dir, f"{idx}.png")
        if os.path.exists(out_path):
            continue
        with open(path) as f:
            prompt = f.read().strip()

        images = []
        for s in range(args.pic_num):
            gen = make_generator(device, settings.seed + s)
            kwargs = settings.call_kwargs()
            kwargs["generator"] = gen
            images.append(call_pipeline(model, prompt, kwargs).images[0])

        horizontal_tile(images).save(out_path)
        print(f"[dpg] {idx} saved")


if __name__ == "__main__":
    main()
