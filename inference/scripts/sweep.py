"""1-D parameter sweep for a single method across a small prompt set.

Load the method's pipeline once, then re-sample the same prompts under each
candidate value of a single generation-param (e.g. ``guidance_scale``,
``tcfg_rank``, ``oseg_scale``). Output layout plays nicely with
``scripts/make_grid.py``::

    out_dir/{param}={value}/{idx:03d}.png     + {idx:03d}.txt

Usage:
    scripts/sweep.py generation/configs/sd35/cfgpp.py \\
        --param guidance_scale --values 2.0 3.0 4.0 5.5 \\
        --prompts prompts_sweep.txt \\
        --out-dir outputs/sweep/cfgpp_scale
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.common import (  # noqa: E402
    IMAGE_LEVEL_KEYS,
    call_pipeline,
    load_config,
    load_pipeline,
    make_generator,
    pick_device,
    resolve_settings,
    write_prompt_sidecar,
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("config", help="Method config to sweep over")
    p.add_argument("--param", required=True,
                   help="Generation-param key to vary (e.g. guidance_scale, tcfg_rank)")
    p.add_argument("--values", nargs="+", required=True, help="Values to try (parsed as float first, else str)")
    p.add_argument("--prompts", required=True,
                   help="Prompt source: .txt (one per line), .jsonl (prompt/caption field), or .csv (prompt_en col)")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--seed", type=int, default=228)
    p.add_argument("--n-prompts", type=int, default=None,
                   help="Cap number of prompts read from --prompts")
    return p.parse_args()


def parse_value(v):
    try:
        if "." in v or "e" in v.lower():
            return float(v)
        return int(v)
    except ValueError:
        return v


def load_prompts(path, cap):
    p = Path(path)
    if p.suffix == ".txt":
        prompts = [ln.strip() for ln in p.read_text().splitlines() if ln.strip()]
    elif p.suffix == ".jsonl":
        import json
        prompts = []
        for line in p.read_text().splitlines():
            if not line.strip():
                continue
            obj = json.loads(line)
            prompts.append(obj.get("prompt") or obj.get("caption") or obj.get("prompt_en"))
    elif p.suffix == ".csv":
        import pandas as pd
        df = pd.read_csv(p, dtype=str)
        col = "prompt_en" if "prompt_en" in df.columns else "prompt"
        prompts = df[col].tolist()
    else:
        sys.exit(f"Unsupported prompts file: {p}")
    if cap:
        prompts = prompts[:cap]
    return prompts


def main():
    args = parse_args()
    values = [parse_value(v) for v in args.values]
    prompts = load_prompts(args.prompts, args.n_prompts)

    pipeline_spec, gen_params = load_config(args.config)
    base_settings = resolve_settings(gen_params, {"seed": args.seed})

    device = pick_device()
    model = load_pipeline(pipeline_spec, device)
    print(f"[sweep] config={args.config} param={args.param} values={values} prompts={len(prompts)}")

    for val in values:
        sub = Path(args.out_dir) / f"{args.param}={val}"
        sub.mkdir(parents=True, exist_ok=True)

        settings = resolve_settings(gen_params, {"seed": args.seed, args.param: val})
        # If the swept key isn't a GenSettings field, resolve_settings stashed it in extra.
        # Either way settings.call_kwargs already reflects it.

        for idx, prompt in enumerate(prompts):
            out_path = sub / f"{idx:03d}.png"
            if out_path.exists():
                continue
            kwargs = settings.call_kwargs()
            kwargs["generator"] = make_generator(device, base_settings.seed + idx)
            img = call_pipeline(model, prompt, kwargs).images[0]
            img.save(out_path)
            write_prompt_sidecar(str(out_path), prompt)
            print(f"[{args.param}={val}] {idx:03d}/{len(prompts)-1}")


if __name__ == "__main__":
    main()
