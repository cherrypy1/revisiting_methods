"""SD3.5 with pluggable guidance methods.
Single pipeline class `StableDiffusion3MethodsPipeline` that dispatches by `method`:
  - cfg: vanilla classifier-free guidance (paper: arXiv:2207.12598)
  - no_cfg: drop the conditional update entirely
  - cfgpp: CFG++ (arXiv:2406.08070)
  - cfg0s: CFG-Zero* (arXiv:2503.18886)
  - apg: Adaptive Projected Guidance (arXiv:2410.02416)
  - tcfg: Tangential Damping CFG (arXiv:2503.18137)
  - sag: Self-Attention Guidance (arXiv:2210.00939)
  - seg: Smoothed Energy Guidance (arXiv:2408.00760)
  - oseg: Orthogonal SEG (WACV2026 Fahim)
The diffusers built-in ``StableDiffusion3PAGPipeline`` already supports it cleanly, so the PAG factory
just uses that upstream class.
"""

from __future__ import annotations

import math
from typing import Any, Callable, Dict, List, Optional, Union

import torch
import torch.nn.functional as F

from diffusers.models.attention_processor import Attention, JointAttnProcessor2_0
from diffusers.utils import is_torch_xla_available, logging
from diffusers.pipelines.stable_diffusion_3.pipeline_stable_diffusion_3 import StableDiffusion3Pipeline


if is_torch_xla_available():
    import torch_xla.core.xla_model as xm  # noqa
    XLA_AVAILABLE = True
else:
    XLA_AVAILABLE = False


logger = logging.get_logger(__name__)


# Guidance combinators

def _flatten(t: torch.Tensor) -> torch.Tensor:
    return t.reshape(t.shape[0], -1)


def _project(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-8):
    """Decompose `a` into components parallel / orthogonal to `b` (per-sample).

    Dot-products are done in fp32: the flattened latent has ~16k+ elements, so
    summing squares in fp16 overflows (>65504) and gives a garbage coefficient.
    """
    af, bf = _flatten(a).float(), _flatten(b).float()
    coef = (af * bf).sum(dim=-1, keepdim=True) / (bf * bf).sum(dim=-1, keepdim=True).clamp(min=eps)
    parallel = coef.view(-1, *([1] * (a.ndim - 1))).to(a.dtype) * b
    return a - parallel, parallel  # (orthogonal, parallel)


def apg_combine(pred_cond, pred_uncond, scale, eta, step_radius, momentum, prev_delta):
    """Adaptive Projected Guidance: clip-then-decompose, eta on orthogonal part."""
    delta = pred_cond - pred_uncond
    if momentum != 0.0 and prev_delta is not None:
        delta = delta + momentum * prev_delta
    new_prev = delta.detach()
    if step_radius is not None and step_radius > 0:
        norms = _flatten(delta).float().norm(dim=-1, keepdim=True).clamp(min=1e-8)  # fp32: fp16 norm overflows
        scale_clip = (norms.clamp(max=step_radius) / norms).view(-1, *([1] * (delta.ndim - 1))).to(delta.dtype)
        delta = delta * scale_clip
    orth, parallel = _project(delta, pred_cond)
    # APG keeps the orthogonal component and DOWN-weights the parallel one (eta<1),
    # because the parallel-to-prediction part is what drives oversaturation. The two
    # were swapped here (parallel kept, orth down-weighted) → with eta=0 the guidance
    # collapsed to the small parallel coef, giving weak/muddy, param-insensitive output.
    # Weight is (scale-1) to match CFG/diffusers normalized_guidance (scale=1 => no guidance).
    return pred_cond + (scale - 1.0) * (orth + eta * parallel), new_prev


def tcfg_combine(pred_cond, pred_uncond, scale, rank: int = 1):
    """Tangential Damping CFG: project uncond onto top-`rank` SVD directions."""
    out_dtype = pred_uncond.dtype
    batch = pred_cond.shape[0]
    stacked = torch.stack((pred_cond, pred_uncond), dim=1).to(dtype=torch.float32)
    stacked_flat = stacked.reshape(batch, 2, -1)
    _, _, vh = torch.linalg.svd(stacked_flat, full_matrices=False)
    keep = max(1, min(int(rank), vh.shape[1]))
    vh_keep = vh.clone()
    vh_keep[:, keep:] = 0.0
    uncond_flat = pred_uncond.to(dtype=torch.float32).reshape(batch, 1, -1)
    projected = (uncond_flat @ vh.transpose(-2, -1)) @ vh_keep
    pred_uncond_tan = projected.reshape_as(pred_uncond).to(dtype=out_dtype)
    # CFG with the filtered uncond as base (paper: s_hat + w*(cond - s_hat)); reduces to
    # plain CFG when uncond_tan == uncond (rank>=2). Was `pred_cond + scale*(...)` which
    # used the wrong base (effective scale+1, oversaturated).
    return pred_uncond_tan + scale * (pred_cond - pred_uncond_tan)


def cfg0s_combine(pred_cond, pred_uncond, scale, current_step, zero_steps, use_zero_init):
    """CFG-Zero*: optimal alpha + early zero-init.

    Optimal alpha minimises ||pred_cond - alpha * pred_uncond||^2 per sample.
    """
    if use_zero_init and current_step <= zero_steps:
        return torch.zeros_like(pred_cond)
    # fp32: the optimal-alpha dot/norm over the flattened latent overflows in fp16
    # (>65504), which sends alpha to garbage and the trajectory to a black image.
    pf, uf = _flatten(pred_cond).float(), _flatten(pred_uncond).float()
    dot = (pf * uf).sum(dim=-1)
    u2 = (uf * uf).sum(dim=-1).clamp(min=1e-8)
    alpha = (dot / u2).view(-1, *([1] * (pred_cond.ndim - 1))).to(pred_cond.dtype)
    pred_uncond_opt = alpha * pred_uncond
    return pred_uncond_opt + scale * (pred_cond - pred_uncond_opt)


# Attention processors for SAG / SEG / OSEG. Each is batch-position-aware:
# the pipeline concatenates [uncond, cond, perturb] (or [cond, perturb] when
# no CFG) in one transformer call; the proc only mutates the perturb chunk.

def _gaussian_blur_2d(img: torch.Tensor, kernel_size: int, sigma: float) -> torch.Tensor:
    """Separable 2D Gaussian blur over the last two dims."""
    height = img.shape[-1]
    kernel_size = min(kernel_size, height - (height % 2 - 1))
    half = (kernel_size - 1) * 0.5
    x = torch.linspace(-half, half, steps=kernel_size, device=img.device, dtype=img.dtype)
    pdf = torch.exp(-0.5 * (x / sigma).pow(2))
    k1d = pdf / pdf.sum()
    k2d = torch.mm(k1d[:, None], k1d[None, :])
    k2d = k2d.expand(img.shape[-3], 1, k2d.shape[0], k2d.shape[1])
    pad = [kernel_size // 2] * 4
    img = F.pad(img, pad, mode="reflect")
    return F.conv2d(img, k2d, groups=img.shape[-3])


class SEGCFGSelfAttnProcessor(JointAttnProcessor2_0):
    """SEG joint-attn proc: gaussian-blur the perturb-chunk queries before SDPA."""

    def __init__(self, blur_sigma: float = 1.0, do_cfg: bool = True,
                 inf_blur_threshold: float = 9999.0, orthogonal: bool = False):
        self.blur_sigma = blur_sigma
        self.do_cfg = do_cfg
        self.inf_blur = blur_sigma > inf_blur_threshold
        # OSEG (orthogonal=True): blur ONLY the component of the query orthogonal to its
        # per-token key, keeping the parallel part — vs SEG which blurs the whole query.
        self.orthogonal = orthogonal

    def __call__(self, attn: Attention, hidden_states, encoder_hidden_states=None,
                 attention_mask=None, *args, **kwargs):
        residual = hidden_states
        batch_size = hidden_states.shape[0]

        query = attn.to_q(hidden_states)
        key = attn.to_k(hidden_states)
        value = attn.to_v(hidden_states)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads
        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)

        spatial_tokens = query.shape[2]
        height = width = math.isqrt(spatial_tokens)
        can_reshape_square = height * width == spatial_tokens

        chunks = 3 if self.do_cfg else 2
        n_chunk = batch_size // chunks
        q_split = list(query.chunk(chunks))
        q_ptb = q_split[-1]
        if can_reshape_square:
            if self.orthogonal:
                # decompose q into parallel/orthogonal w.r.t. its per-token key; blur only orth
                k_ptb = key.chunk(chunks)[-1]
                k_norm = torch.nn.functional.normalize(k_ptb.float(), dim=-1)
                q_par = (q_ptb.float() * k_norm).sum(dim=-1, keepdim=True) * k_norm
                target = (q_ptb.float() - q_par).to(q_ptb.dtype)
            else:
                q_par, target = None, q_ptb
            t = target.permute(0, 1, 3, 2).reshape(n_chunk, attn.heads * head_dim, height, width)
            if self.inf_blur:
                t = t.mean(dim=(-2, -1), keepdim=True).expand(-1, -1, height, width)
            else:
                ks = math.ceil(6 * self.blur_sigma) + 1 - math.ceil(6 * self.blur_sigma) % 2
                t = _gaussian_blur_2d(t, ks, self.blur_sigma)
            t = t.reshape(n_chunk, attn.heads, head_dim, height * width).permute(0, 1, 3, 2)
            q_ptb = (q_par.to(q_ptb.dtype) + t) if self.orthogonal else t
        q_split[-1] = q_ptb
        query = torch.cat(q_split, dim=0)

        if encoder_hidden_states is not None:
            ehs_q = attn.add_q_proj(encoder_hidden_states).view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
            ehs_k = attn.add_k_proj(encoder_hidden_states).view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
            ehs_v = attn.add_v_proj(encoder_hidden_states).view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
            if attn.norm_added_q is not None:
                ehs_q = attn.norm_added_q(ehs_q)
            if attn.norm_added_k is not None:
                ehs_k = attn.norm_added_k(ehs_k)
            query = torch.cat([query, ehs_q], dim=2)
            key = torch.cat([key, ehs_k], dim=2)
            value = torch.cat([value, ehs_v], dim=2)

        hidden_states = F.scaled_dot_product_attention(query, key, value, dropout_p=0.0, is_causal=False)
        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim).to(query.dtype)

        if encoder_hidden_states is not None:
            hidden_states, encoder_hidden_states = (
                hidden_states[:, : residual.shape[1]],
                hidden_states[:, residual.shape[1]:],
            )
            if not attn.context_pre_only:
                encoder_hidden_states = attn.to_add_out(encoder_hidden_states)

        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)
        if encoder_hidden_states is not None:
            return hidden_states, encoder_hidden_states
        return hidden_states


class SAGCFGSelfAttnProcessor(JointAttnProcessor2_0):
    """SAG joint-attn proc: standard SDPA but also stores attention probs on the
    perturb chunk so the pipeline can mask high-attn regions for the SAG pass."""

    def __init__(self, do_cfg: bool = True, blur_sigma: float = 2.0):
        self.do_cfg = do_cfg
        self.blur_sigma = blur_sigma
        self.attention_probs: Optional[torch.Tensor] = None

    def __call__(self, attn: Attention, hidden_states, encoder_hidden_states=None,
                 attention_mask=None, *args, **kwargs):
        residual = hidden_states
        batch_size = hidden_states.shape[0]

        query = attn.to_q(hidden_states)
        key = attn.to_k(hidden_states)
        value = attn.to_v(hidden_states)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads
        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)

        if encoder_hidden_states is not None:
            ehs_q = attn.add_q_proj(encoder_hidden_states).view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
            ehs_k = attn.add_k_proj(encoder_hidden_states).view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
            ehs_v = attn.add_v_proj(encoder_hidden_states).view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
            if attn.norm_added_q is not None:
                ehs_q = attn.norm_added_q(ehs_q)
            if attn.norm_added_k is not None:
                ehs_k = attn.norm_added_k(ehs_k)
            query = torch.cat([query, ehs_q], dim=2)
            key = torch.cat([key, ehs_k], dim=2)
            value = torch.cat([value, ehs_v], dim=2)

        # Hand-rolled SDPA so we can perturb the perturb chunk's self-attended values.
        scale = head_dim ** -0.5
        attention_scores = torch.matmul(query, key.transpose(-1, -2)) * scale
        attention_probs = attention_scores.softmax(dim=-1)

        # SAG perturbation: blur the perturb-chunk image-token VALUES in high
        # self-attention regions, so its prediction degrades there. The pipeline then
        # guides via np_text - np_ptb. Without this the perturb chunk equals the cond
        # chunk (np_ptb == np_text) and sag_scale has nothing to act on.
        sample_tokens = residual.shape[1]
        chunks = 3 if self.do_cfg else 2
        n_chunk = batch_size // chunks
        side = int(round(sample_tokens ** 0.5))
        if n_chunk > 0 and side * side == sample_tokens:
            sal = attention_probs[-n_chunk:, :, :sample_tokens, :sample_tokens].mean(dim=1).mean(dim=1)
            sal = sal.view(n_chunk, 1, side, side).float()
            lo = sal.amin(dim=(-2, -1), keepdim=True)
            hi = sal.amax(dim=(-2, -1), keepdim=True)
            mask = ((sal - lo) / (hi - lo).clamp(min=1e-6)).to(value.dtype)   # (n,1,H,W) in [0,1]
            v_img = value[-n_chunk:, :, :sample_tokens, :]                     # (n,heads,T,hd)
            heads_, hd_ = v_img.shape[1], v_img.shape[3]
            vi = v_img.permute(0, 1, 3, 2).reshape(n_chunk, heads_ * hd_, side, side)
            ks = max(3, 2 * int(2 * self.blur_sigma) + 1)
            vi_blur = _gaussian_blur_2d(vi, ks, self.blur_sigma)
            vi_new = vi * (1 - mask) + vi_blur * mask
            v_img_new = vi_new.reshape(n_chunk, heads_, hd_, sample_tokens).permute(0, 1, 3, 2)
            value = value.clone()
            value[-n_chunk:, :, :sample_tokens, :] = v_img_new.to(value.dtype)
        self.attention_probs = attention_probs[-n_chunk:, :, :sample_tokens, :sample_tokens].flatten(0, 1).detach()

        hidden_states = attention_probs @ value
        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim).to(query.dtype)

        if encoder_hidden_states is not None:
            hidden_states, encoder_hidden_states = (
                hidden_states[:, : residual.shape[1]],
                hidden_states[:, residual.shape[1]:],
            )
            if not attn.context_pre_only:
                encoder_hidden_states = attn.to_add_out(encoder_hidden_states)

        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)
        if encoder_hidden_states is not None:
            return hidden_states, encoder_hidden_states
        return hidden_states


def _expand_seg_applied_layers(layers: Optional[List[str]]) -> Optional[set]:
    """Expand range syntax (e.g. ``d0..12``) into explicit tags (``d0`` … ``d12``)."""
    if layers is None:
        return None
    out: set[str] = set()
    for name in layers:
        if ".." in name and len(name) >= 2:
            prefix = name[0]
            start_end = name[1:].split("..", 1)
            if len(start_end) != 2:
                continue
            try:
                start = int(start_end[0])
                end = int(start_end[1])
            except ValueError:
                continue
            step = 1 if end >= start else -1
            for i in range(start, end + step, step):
                out.add(f"{prefix}{i}")
        else:
            out.add(name)
    return out


# Pipeline

class StableDiffusion3MethodsPipeline(StableDiffusion3Pipeline):
    """SD3.5-medium with a ``method`` kwarg selecting the guidance strategy.

    All upstream features that aren't method-specific (encode_prompt,
    prepare_latents, scheduler setup, VAE decode) are inherited unchanged.
    Only ``__call__`` is overridden to dispatch the denoising loop.
    """

    @property
    def do_seg(self):
        return getattr(self, "_seg_scale", 0.0) > 0 or getattr(self, "_oseg_scale", 0.0) > 0

    @property
    def do_sag(self):
        return getattr(self, "_sag_scale", 0.0) > 0

    def _install_seg_procs(self, blur_sigma: float, applied_layers: Optional[List[str]],
                           do_cfg: bool, orthogonal: bool = False):
        """Set SEGCFGSelfAttnProcessor on selected joint-attn blocks (SEG or, with
        ``orthogonal=True``, OSEG). Returns the original processors dict to restore.
        """
        orig = self.transformer.attn_processors.copy()
        procs = orig.copy()
        expanded = _expand_seg_applied_layers(applied_layers)
        replace = SEGCFGSelfAttnProcessor(blur_sigma=blur_sigma, do_cfg=do_cfg,
                                          orthogonal=orthogonal)
        for name in list(procs.keys()):
            if expanded is None:
                procs[name] = replace
                continue
            parts = name.split(".")
            if len(parts) < 2:
                continue
            try:
                block_num = int(parts[1])
            except ValueError:
                continue
            # SD3.5 has dual attention on blocks 0..12 (joint) + self-only on 13..23.
            tag = f"d{block_num}" if block_num <= 12 else f"s{block_num - 13}"
            if tag in expanded:
                procs[name] = replace
        self.transformer.set_attn_processor(procs)
        return orig

    def _install_sag_procs(self, applied_layers: Optional[List[str]], do_cfg: bool):
        orig = self.transformer.attn_processors.copy()
        procs = orig.copy()
        expanded = _expand_seg_applied_layers(applied_layers)
        replace = SAGCFGSelfAttnProcessor(do_cfg=do_cfg)
        chosen = []
        for name in list(procs.keys()):
            if expanded is None:
                procs[name] = replace
                chosen.append(name)
                continue
            parts = name.split(".")
            if len(parts) < 2:
                continue
            try:
                block_num = int(parts[1])
            except ValueError:
                continue
            tag = f"d{block_num}" if block_num <= 12 else f"s{block_num - 13}"
            if tag in expanded:
                procs[name] = replace
                chosen.append(name)
        self.transformer.set_attn_processor(procs)
        return orig, replace

    @torch.no_grad()
    def __call__(
        self,
        prompt: Union[str, List[str]] = None,
        prompt_2: Optional[Union[str, List[str]]] = None,
        prompt_3: Optional[Union[str, List[str]]] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 28,
        guidance_scale: float = 7.0,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        negative_prompt_2: Optional[Union[str, List[str]]] = None,
        negative_prompt_3: Optional[Union[str, List[str]]] = None,
        num_images_per_prompt: Optional[int] = 1,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        joint_attention_kwargs: Optional[Dict[str, Any]] = None,
        clip_skip: Optional[int] = None,
        max_sequence_length: int = 256,
        mu: Optional[float] = None,
        # method dispatch
        method: str = "cfg",
        # cfgpp
        cfgpp_w: Optional[float] = None,  # if None, falls back to guidance_scale
        # cfg0s
        use_zero_init: bool = True,
        zero_steps: int = 0,
        # apg
        apg_eta: float = 0.0,
        apg_step_radius: float = 15.0,
        apg_momentum: float = -0.5,
        # tcfg
        tcfg_rank: int = 1,
        # seg/oseg/sag
        seg_scale: float = 0.0,
        seg_blur_sigma: float = 10.0,
        seg_applied_layers: Optional[List[str]] = None,
        oseg_scale: float = 0.0,
        sag_scale: float = 0.0,
        sag_applied_layers: Optional[List[str]] = None,
        **_unused,
    ):
        # Resolve method aliases / state
        method = method.lower()
        height = height or self.default_sample_size * self.vae_scale_factor
        width = width or self.default_sample_size * self.vae_scale_factor

        self._guidance_scale = guidance_scale
        self._clip_skip = clip_skip
        self._joint_attention_kwargs = joint_attention_kwargs
        self._interrupt = False
        self._seg_scale = seg_scale
        self._oseg_scale = oseg_scale
        self._sag_scale = sag_scale

        # cfgpp's guidance_scale IS the CFG++ weight w (lambda in [0,1]); its useful
        # range is w<=1, so it must NOT be gated by the >1 CFG threshold — it always
        # needs the uncond+cond pair. (Without this, w<=1 fell through to a single
        # conditional forward => looked like no guidance.)
        do_cfg = method != "no_cfg" and (guidance_scale > 1.0 or method == "cfgpp")
        do_seg = (seg_scale + oseg_scale) > 0 and method in ("seg", "oseg")
        do_sag = sag_scale > 0 and method == "sag"

        if isinstance(prompt, str):
            batch_size = 1
        elif isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]
        device = self._execution_device

        # Encode prompts (uses upstream encode_prompt with all 3 text-encoders).
        lora_scale = (
            self.joint_attention_kwargs.get("scale", None) if self.joint_attention_kwargs else None
        )
        (
            prompt_embeds, negative_prompt_embeds, pooled_prompt_embeds, negative_pooled_prompt_embeds,
        ) = self.encode_prompt(
            prompt=prompt, prompt_2=prompt_2, prompt_3=prompt_3,
            negative_prompt=negative_prompt, negative_prompt_2=negative_prompt_2,
            negative_prompt_3=negative_prompt_3,
            do_classifier_free_guidance=do_cfg,
            prompt_embeds=prompt_embeds, negative_prompt_embeds=negative_prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
            device=device, clip_skip=self.clip_skip,
            num_images_per_prompt=num_images_per_prompt,
            max_sequence_length=max_sequence_length,
            lora_scale=lora_scale,
        )

        # Build the batched embeddings — chunks change with do_seg / do_sag.
        if do_seg or do_sag:
            if do_cfg:
                prompt_embeds_in = torch.cat([negative_prompt_embeds, prompt_embeds, prompt_embeds], dim=0)
                pooled_in = torch.cat([
                    negative_pooled_prompt_embeds, pooled_prompt_embeds, pooled_prompt_embeds,
                ], dim=0)
            else:
                prompt_embeds_in = torch.cat([prompt_embeds, prompt_embeds], dim=0)
                pooled_in = torch.cat([pooled_prompt_embeds, pooled_prompt_embeds], dim=0)
        elif do_cfg:
            prompt_embeds_in = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
            pooled_in = torch.cat([negative_pooled_prompt_embeds, pooled_prompt_embeds], dim=0)
        else:
            prompt_embeds_in = prompt_embeds
            pooled_in = pooled_prompt_embeds

        # Latents
        num_channels_latents = self.transformer.config.in_channels
        latents = self.prepare_latents(
            batch_size * num_images_per_prompt, num_channels_latents,
            height, width, prompt_embeds_in.dtype, device, generator, latents,
        )

        # Timesteps (uses upstream helper).
        from diffusers.pipelines.stable_diffusion_3.pipeline_stable_diffusion_3 import (
            calculate_shift, retrieve_timesteps,
        )
        scheduler_kwargs: Dict[str, Any] = {}
        if self.scheduler.config.get("use_dynamic_shifting", None) and mu is None:
            _, _, h_lat, w_lat = latents.shape
            image_seq_len = (h_lat // self.transformer.config.patch_size) * (
                w_lat // self.transformer.config.patch_size
            )
            mu = calculate_shift(
                image_seq_len,
                self.scheduler.config.get("base_image_seq_len", 256),
                self.scheduler.config.get("max_image_seq_len", 4096),
                self.scheduler.config.get("base_shift", 0.5),
                self.scheduler.config.get("max_shift", 1.16),
            )
            scheduler_kwargs["mu"] = mu
        elif mu is not None:
            scheduler_kwargs["mu"] = mu

        timesteps, num_inference_steps = retrieve_timesteps(
            self.scheduler, num_inference_steps, device, sigmas=None, **scheduler_kwargs,
        )
        num_warmup_steps = max(len(timesteps) - num_inference_steps * self.scheduler.order, 0)
        self._num_timesteps = len(timesteps)

        # Install perturb-attn procs ONCE per __call__ (restore at end).
        orig_attn_procs: Optional[Dict] = None
        sag_proc: Optional[SAGCFGSelfAttnProcessor] = None
        if do_seg:
            orig_attn_procs = self._install_seg_procs(
                seg_blur_sigma, seg_applied_layers, do_cfg, orthogonal=(method == "oseg"))
        elif do_sag:
            orig_attn_procs, sag_proc = self._install_sag_procs(sag_applied_layers, do_cfg)

        apg_prev_delta: Optional[torch.Tensor] = None

        # Denoising loop
        try:
            with self.progress_bar(total=num_inference_steps) as bar:
                for i, t in enumerate(timesteps):
                    if self._interrupt:
                        continue

                    if do_seg or do_sag:
                        chunks = 3 if do_cfg else 2
                        latent_model_input = torch.cat([latents] * chunks)
                    elif do_cfg:
                        latent_model_input = torch.cat([latents] * 2)
                    else:
                        latent_model_input = latents

                    timestep = t.expand(latent_model_input.shape[0])
                    noise_pred = self.transformer(
                        hidden_states=latent_model_input,
                        timestep=timestep,
                        encoder_hidden_states=prompt_embeds_in,
                        pooled_projections=pooled_in,
                        joint_attention_kwargs=self.joint_attention_kwargs,
                        return_dict=False,
                    )[0]

                    cfgpp_target = None
                    if not do_cfg and not (do_seg or do_sag):
                        # no_cfg or pure single-stream — just use what we got
                        noise_pred_combined = noise_pred
                    elif do_seg:
                        if do_cfg:
                            np_uncond, np_text, np_ptb = noise_pred.chunk(3)
                        else:
                            np_text, np_ptb = noise_pred.chunk(2)
                            np_uncond = None
                        cfg_delta = (np_text - np_uncond) if do_cfg else 0.0
                        # SEG: np_ptb = full-query-blurred pred. OSEG: np_ptb = orthogonal-
                        # query-blurred pred (the perturbation differs in the attn proc).
                        seg_delta = np_text - np_ptb
                        if method == "oseg":
                            # perturbation-guidance form: cond + (w-1)*CFG + oseg_scale*(cond - oseg_pred)
                            noise_pred_combined = (
                                np_text
                                + (guidance_scale - 1.0) * cfg_delta
                                + oseg_scale * seg_delta
                            )
                        else:  # seg
                            noise_pred_combined = (
                                np_text
                                + (guidance_scale - 1.0) * cfg_delta
                                + seg_scale * seg_delta
                            )
                    elif do_sag:
                        if do_cfg:
                            np_uncond, np_text, np_ptb = noise_pred.chunk(3)
                            cfg_delta = np_text - np_uncond
                        else:
                            np_text, np_ptb = noise_pred.chunk(2)
                            cfg_delta = 0.0
                            np_uncond = None
                        sag_delta = np_text - np_ptb
                        noise_pred_combined = (
                            np_text
                            + (guidance_scale - 1.0) * cfg_delta
                            + sag_scale * sag_delta
                        )
                    else:
                        # standard CFG-style methods
                        np_uncond, np_text = noise_pred.chunk(2)
                        if method == "cfg":
                            noise_pred_combined = np_uncond + guidance_scale * (np_text - np_uncond)
                        elif method == "cfgpp":
                            w = cfgpp_w if cfgpp_w is not None else guidance_scale
                            # CFG++ (arXiv:2406.08070), flow form. DDIM:
                            #   x_{t-1} = sqrt(a_{t-1}) x0_guided + sqrt(1-a_{t-1}) eps_uncond
                            # rectified-flow: sqrt(a)->(1-sigma), sqrt(1-a)->sigma, eps=x0+v.
                            # KEY: renoise with the UNCONDITIONAL eps, not the guided one.
                            sigma = self.scheduler.sigmas[i].to(latents.device, latents.dtype)
                            sigma_next = (
                                self.scheduler.sigmas[i + 1].to(latents.device, latents.dtype)
                                if i + 1 < len(self.scheduler.sigmas) else sigma * 0.0
                            )
                            v_uncond = np_uncond.to(dtype=torch.float32)
                            v_text = np_text.to(dtype=torch.float32)
                            lat32 = latents.to(dtype=torch.float32)
                            sig32 = sigma.to(dtype=torch.float32)
                            sn32 = sigma_next.to(dtype=torch.float32)
                            x0_uncond = lat32 - sig32 * v_uncond            # x0 = x - sigma*v
                            x0_text = lat32 - sig32 * v_text
                            x0_guided = x0_uncond + w * (x0_text - x0_uncond)
                            eps_uncond = x0_uncond + v_uncond               # eps = x0 + v
                            x_next_cfgpp = (1.0 - sn32) * x0_guided + sn32 * eps_uncond
                            cfgpp_target = (x_next_cfgpp.to(dtype=latents.dtype), sigma, sigma_next)
                            noise_pred_combined = np_uncond  # placeholder (step overridden)
                        elif method == "cfg0s":
                            noise_pred_combined = cfg0s_combine(
                                np_text, np_uncond, guidance_scale, i, zero_steps, use_zero_init,
                            )
                        elif method == "apg":
                            noise_pred_combined, apg_prev_delta = apg_combine(
                                np_text, np_uncond, guidance_scale,
                                apg_eta, apg_step_radius, apg_momentum, apg_prev_delta,
                            )
                        elif method == "tcfg":
                            noise_pred_combined = tcfg_combine(
                                np_text, np_uncond, guidance_scale, tcfg_rank,
                            )
                        elif method == "no_cfg":
                            noise_pred_combined = np_text
                        else:
                            raise ValueError(f"unknown method: {method!r}")

                    # CFG++ overrides the scheduler step.
                    latents_dtype = latents.dtype
                    if cfgpp_target is not None:
                        x_next, sigma, sigma_next = cfgpp_target
                        latents = x_next
                    else:
                        latents = self.scheduler.step(
                            noise_pred_combined, t, latents, return_dict=False,
                        )[0]
                    if latents.dtype != latents_dtype and torch.backends.mps.is_available():
                        latents = latents.to(latents_dtype)

                    if i == len(timesteps) - 1 or (
                        (i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0
                    ):
                        bar.update()
                    if XLA_AVAILABLE:
                        xm.mark_step()

        finally:
            if orig_attn_procs is not None:
                self.transformer.set_attn_processor(orig_attn_procs)

        if output_type == "latent":
            image = latents
        else:
            latents = (latents / self.vae.config.scaling_factor) + self.vae.config.shift_factor
            image = self.vae.decode(latents, return_dict=False)[0]
            image = self.image_processor.postprocess(image, output_type=output_type)

        self.maybe_free_model_hooks()
        from diffusers.pipelines.stable_diffusion_3.pipeline_output import StableDiffusion3PipelineOutput
        if not return_dict:
            return (image,)
        return StableDiffusion3PipelineOutput(images=image)
