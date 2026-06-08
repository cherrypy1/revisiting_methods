"""Cosmos2 HP sweep, single-GPU, no cpu_offload.

Strategy: encode all unique prompts with T5 once, push T5 to CPU, then load
transformer+VAE on GPU and run trials passing prompt_embeds directly.
encode_prompt short-circuits when both prompt_embeds and negative_prompt_embeds
are provided, so T5 is never invoked during diffusion.

Driver designed for parallel launch:
    CUDA_VISIBLE_DEVICES=0 python scripts/cosmos_hp_run.py --methods pag seg oseg sag cfg0s &
    CUDA_VISIBLE_DEVICES=1 python scripts/cosmos_hp_run.py --methods cfg no_cfg cfgpp apg tcfg &
"""
from __future__ import annotations

import sys as _sys
import types as _types
from importlib.machinery import ModuleSpec as _ModuleSpec


# Stub soundfile before transformers import (libsndfile.so unavailable).
# soxr / sentencepiece load fine once gnu14/14.1 module is loaded (libstdc++).
if "soundfile" not in _sys.modules:
    sf = _types.ModuleType("soundfile")
    sf.__spec__ = _ModuleSpec("soundfile", loader=None)
    sf.__version__ = "0.0.0"
    sf.read = lambda *a, **kw: (None, None)
    sf.write = lambda *a, **kw: None
    sf.info = lambda *a, **kw: None
    sf.SoundFile = type("SoundFile", (), {})
    _sys.modules["soundfile"] = sf

import argparse
import gc
import json
import os
import sys
import time
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.common import make_generator, pick_device, write_prompt_sidecar  # noqa: E402
from scripts.cosmos_hp_search import GRIDS, DEFAULTS, stratified_sample, trial_name  # noqa: E402

COSMOS2 = "nvidia/Cosmos-Predict2-2B-Text2Image"
# GenEval prompt set used as the stratified sampling source (external repo).
GENEVAL_ROOT = Path(os.environ.get("GENEVAL_ROOT", str(Path.home() / "geneval-bench")))
DEFAULT_PROMPTS = GENEVAL_ROOT / "prompts" / "evaluation_metadata.jsonl"
DEFAULT_OUT = PROJECT_ROOT / "outputs" / "cosmos2" / "hp"


def encode_all(prompts, device, max_len=512):
    from transformers import T5EncoderModel, T5TokenizerFast
    print(f"[hp] loading T5 for pre-encoding ({len(prompts)} prompts)...")
    t0 = time.time()
    tok = T5TokenizerFast.from_pretrained(COSMOS2, subfolder="tokenizer")
    enc = T5EncoderModel.from_pretrained(
        COSMOS2, subfolder="text_encoder", torch_dtype=torch.bfloat16,
    ).to(device)
    enc.eval()
    cache = {}
    for p in prompts:
        ti = tok([p], padding="max_length", max_length=max_len, truncation=True,
                 return_tensors="pt", return_length=True, return_offsets_mapping=False)
        ids = ti.input_ids.to(device)
        mask = ti.attention_mask.bool().to(device)
        with torch.no_grad():
            emb = enc(ids, attention_mask=mask).last_hidden_state.to(torch.bfloat16)
        lengths = mask.sum(dim=1).cpu()
        for i, length in enumerate(lengths):
            emb[i, length:] = 0
        cache[p] = emb.cpu()
    del enc, tok
    gc.collect()
    torch.cuda.empty_cache()
    print(f"[hp] encoded in {time.time()-t0:.1f}s; cache={len(cache)}")
    return cache


def build_pipe_no_t5(device):
    from diffusers import AutoencoderKLWan, CosmosTransformer3DModel, FlowMatchEulerDiscreteScheduler
    from pipelines.cosmos2.pipeline_cosmos2_methods import Cosmos2MethodsPipeline

    print("[hp] loading transformer + VAE...")
    t0 = time.time()
    transformer = CosmosTransformer3DModel.from_pretrained(
        COSMOS2, subfolder="transformer", torch_dtype=torch.bfloat16,
    ).to(device)
    vae = AutoencoderKLWan.from_pretrained(
        COSMOS2, subfolder="vae", torch_dtype=torch.bfloat16,
    ).to(device)
    scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(COSMOS2, subfolder="scheduler")
    pipe = Cosmos2MethodsPipeline(
        text_encoder=None, tokenizer=None,
        transformer=transformer, vae=vae, scheduler=scheduler, safety_checker=None,
    )
    pipe._execution_device_override = torch.device(device)
    # Patch property: parent uses text_encoder.device. We have none.
    type(pipe)._execution_device = property(lambda self: self._execution_device_override)
    # VAE decode needs tiling on V100 32GB or it OOMs at 1024x1024.
    try:
        pipe.vae.enable_tiling()
    except Exception as e:
        print(f"[hp] vae.enable_tiling failed: {e}")
    try:
        pipe.vae.enable_slicing()
    except Exception:
        pass
    print(f"[hp] pipe ready in {time.time()-t0:.1f}s; "
          f"mem free={torch.cuda.mem_get_info()[0]/1e9:.1f}GB")
    return pipe


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--methods", nargs="+", required=True)
    p.add_argument("--prompts", default=str(DEFAULT_PROMPTS))
    p.add_argument("--out-root", default=str(DEFAULT_OUT))
    p.add_argument("--per-tag", type=int, default=2)
    p.add_argument("--seed", type=int, default=228)
    p.add_argument("--sample-seed", type=int, default=0)
    p.add_argument("--steps", type=int, default=20)
    p.add_argument("--resolution", type=int, default=1024)
    return p.parse_args()


def main():
    args = parse_args()
    device = pick_device()
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    with open(args.prompts) as fp:
        metadatas = [json.loads(line) for line in fp]
    chosen = stratified_sample(metadatas, args.per_tag, args.sample_seed)
    print(f"[hp] device={device} methods={args.methods} "
          f"prompts/trial={len(chosen)} steps={args.steps}")

    neg = DEFAULTS["negative_prompt"]
    unique = sorted({m["prompt"] for m in chosen} | {neg})
    embed_cache = encode_all(unique, device)
    neg_emb = embed_cache[neg]

    pipe = build_pipe_no_t5(device)

    for method in args.methods:
        if method not in GRIDS:
            print(f"[hp] skip {method}")
            continue
        for trial in GRIDS[method]:
            tname = trial_name(trial)
            trial_root = out_root / method / tname
            trial_root.mkdir(parents=True, exist_ok=True)
            kwargs_tpl = {
                **DEFAULTS, "method": method,
                "num_inference_steps": args.steps,
                "height": args.resolution, "width": args.resolution,
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
                cond = embed_cache[meta["prompt"]].to(device)
                kw = dict(kwargs_tpl)
                kw["generator"] = make_generator(device, args.seed)
                kw["prompt_embeds"] = cond
                kw["negative_prompt_embeds"] = neg_emb.to(device)
                with torch.no_grad():
                    image = pipe(prompt=meta["prompt"], **kw).images[0]
                image.save(img_path)
                write_prompt_sidecar(str(img_path), meta["prompt"])
                torch.cuda.empty_cache()
            print(f"[hp] {method}/{tname} done in {time.time()-t_start:.1f}s")
    print("[hp] ALL DONE")


if __name__ == "__main__":
    main()
