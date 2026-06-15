"""Shared loader for Cosmos-Predict2-2B-Text2Image pipeline.

Strategy: load text_encoder (T5-11B) in 4-bit nf4 via bnb to fit V100 32GB,
load transformer + VAE in bf16, skip NVIDIA safety guardrail.
"""

from __future__ import annotations

import torch
from transformers import T5EncoderModel
from diffusers import AutoencoderKLWan, CosmosTransformer3DModel, FlowMatchEulerDiscreteScheduler
from transformers import T5TokenizerFast


COSMOS2 = "nvidia/Cosmos-Predict2-2B-Text2Image"


def _quant_config():
    try:
        from transformers import BitsAndBytesConfig
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
    except Exception:
        return None


def load_pipeline(method: str, device: str):
    from .pipeline_cosmos2_methods import Cosmos2MethodsPipeline

    qc = _quant_config()
    # device_map pins the 4-bit weights straight onto the GPU at load time.
    # Without it, transformers auto-dispatch calls ``model.to(device)``, which
    # bitsandbytes < 0.43.2 refuses for 4-bit models ("Calling `to()` is not
    # supported for `4-bit` quantized models").
    te_kwargs = dict(
        subfolder="text_encoder",
        quantization_config=qc, torch_dtype=torch.bfloat16,
    )
    if qc is not None:
        te_kwargs["device_map"] = {"": 0}
    text_encoder = T5EncoderModel.from_pretrained(COSMOS2, **te_kwargs)
    tokenizer = T5TokenizerFast.from_pretrained(COSMOS2, subfolder="tokenizer")
    transformer = CosmosTransformer3DModel.from_pretrained(
        COSMOS2, subfolder="transformer", torch_dtype=torch.bfloat16,
    )
    vae = AutoencoderKLWan.from_pretrained(
        COSMOS2, subfolder="vae", torch_dtype=torch.bfloat16,
    )
    scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(COSMOS2, subfolder="scheduler")

    pipe = Cosmos2MethodsPipeline(
        text_encoder=text_encoder, tokenizer=tokenizer,
        transformer=transformer, vae=vae, scheduler=scheduler, safety_checker=None,
    )
    # bnb-quantized text_encoder is already on GPU; move others.
    pipe.transformer.to(device)
    pipe.vae.to(device)
    return pipe
