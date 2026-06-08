"""Cosmos2-Predict2-2B Text2Image with pluggable guidance methods.

Single pipeline class `Cosmos2MethodsPipeline` that dispatches by `method`:
  - cfg: vanilla classifier-free guidance (paper: arXiv:2207.12598)
  - no_cfg: drop the conditional update entirely
  - cfgpp: CFG++ (arXiv:2406.08070) with reparametrized w
  - cfg0s: CFG-Zero* (arXiv:2503.18886) with optimal scale + zero-init
  - apg: Adaptive Projected Guidance (arXiv:2410.02416)
  - tcfg: Tangential Damping CFG (arXiv:2503.18137)
  - sag: Self-Attention Guidance (arXiv:2210.00939)
  - pag: Perturbed-Attention Guidance (arXiv:2403.17377) - identity self-attn
  - seg: Smoothed Energy Guidance (arXiv:2408.00760) - blurred query SDPA
  - oseg: Orthogonal SEG (WACV2026 Fahim) - SEG with orthogonal projection
"""

from __future__ import annotations

import inspect
import math
from typing import Callable, Dict, List, Optional, Union

import numpy as np
import torch
import torch.nn.functional as F

from diffusers.pipelines.cosmos.pipeline_cosmos2_text2image import (
    Cosmos2TextToImagePipeline,
    retrieve_timesteps,
)
from diffusers.pipelines.cosmos.pipeline_output import CosmosImagePipelineOutput
from diffusers.utils import is_torch_xla_available, logging

if is_torch_xla_available():
    import torch_xla.core.xla_model as xm
    XLA_AVAILABLE = True
else:
    XLA_AVAILABLE = False

logger = logging.get_logger(__name__)


# ---------------------------------------------------------------------------
# Guidance combinators (operate on EDM-space x0-style predictions)
# ---------------------------------------------------------------------------

def _flatten_per_sample(t: torch.Tensor) -> torch.Tensor:
    return t.reshape(t.shape[0], -1)


def project_parallel(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-8):
    """Decompose `a` into components parallel & orthogonal to `b` (per-sample)."""
    af, bf = _flatten_per_sample(a), _flatten_per_sample(b)
    dot = (af * bf).sum(dim=-1, keepdim=True)
    bnorm2 = (bf * bf).sum(dim=-1, keepdim=True).clamp(min=eps)
    coef = dot / bnorm2
    parallel = (coef.view(-1, *([1] * (a.ndim - 1)))) * b
    orthogonal = a - parallel
    return parallel, orthogonal


def apg_combine(pred_cond, pred_uncond, scale, eta, step_radius, momentum, prev_delta):
    """Adaptive Projected Guidance (Sadat et al. 2024).
    delta = pred_cond - pred_uncond
    delta + momentum * prev_delta -> norm-clipped to step_radius
    decompose into parallel/orth to pred_cond
    new = pred_cond + scale * (parallel + eta * orth)
    """
    delta = pred_cond - pred_uncond
    if momentum != 0.0 and prev_delta is not None:
        delta = delta + momentum * prev_delta
    new_prev = delta.detach()
    if step_radius is not None and step_radius > 0:
        d_flat = _flatten_per_sample(delta)
        norms = d_flat.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        scale_clip = (norms.clamp(max=step_radius) / norms).view(-1, *([1] * (delta.ndim - 1)))
        delta = delta * scale_clip
    parallel, orth = project_parallel(delta, pred_cond)
    out = pred_cond + scale * (parallel + eta * orth)
    return out, new_prev


def tcfg_combine(pred_cond, pred_uncond, scale, rank: int = 1):
    """Tangential Damping CFG (arXiv:2503.18137 Alg. 1).

    Stack [pred_cond, pred_uncond] into a (B, 2, N) matrix, SVD, project the
    uncond branch onto the top-`rank` right singular vectors, then apply CFG
    with the projected uncond.
    """
    out_dtype = pred_uncond.dtype
    batch_size = pred_cond.shape[0]
    stacked = torch.stack((pred_cond, pred_uncond), dim=1).to(dtype=torch.float32)
    stacked_flat = stacked.reshape(batch_size, 2, -1)
    _, _, vh = torch.linalg.svd(stacked_flat, full_matrices=False)
    keep_rank = max(1, min(int(rank), vh.shape[1]))
    vh_keep = vh.clone()
    vh_keep[:, keep_rank:] = 0.0
    uncond_flat = pred_uncond.to(dtype=torch.float32).reshape(batch_size, 1, -1)
    projected = (uncond_flat @ vh.transpose(-2, -1)) @ vh_keep
    pred_uncond_tan = projected.reshape_as(pred_uncond).to(dtype=out_dtype)
    return pred_cond + scale * (pred_cond - pred_uncond_tan)


def cfg0s_combine(pred_cond, pred_uncond, scale, current_step, total_steps, zero_steps, use_zero_init):
    """CFG-Zero*: optimal-scale + early zero-init.
    optimal alpha minimizes ||pred_cond - alpha*pred_uncond||^2 per sample.
    """
    if use_zero_init and current_step <= zero_steps:
        return torch.zeros_like(pred_cond)
    pf = _flatten_per_sample(pred_cond)
    uf = _flatten_per_sample(pred_uncond)
    dot = (pf * uf).sum(dim=-1)
    u2 = (uf * uf).sum(dim=-1).clamp(min=1e-8)
    alpha = (dot / u2).view(-1, *([1] * (pred_cond.ndim - 1)))
    pred_uncond_opt = alpha * pred_uncond
    return pred_uncond_opt + scale * (pred_cond - pred_uncond_opt)


def cfgpp_x_next(latents, pred_cond, pred_uncond, w, current_sigma, sigma_next):
    """CFG++ (arXiv:2406.08070), EDM-style adaptation for Cosmos2.

    Standard model parametrisation in this pipeline: ``pred`` is the x0
    estimate, and a single Euler step in sigma updates ``x_next = x0 +
    sigma_next * eps``. CFG++ mixes a CFG-guided x0 with an
    *unconditional* noise direction::

        x_next = x0_guided + sigma_next * eps_uncond,
        x0_guided  = pred_cond + (w - 1) * (pred_cond - pred_uncond)
        eps_uncond = (latents - pred_uncond) / current_sigma
    """
    x0_guided = pred_cond + (w - 1.0) * (pred_cond - pred_uncond)
    eps_uncond = (latents - pred_uncond) / current_sigma
    return x0_guided + sigma_next * eps_uncond


# ---------------------------------------------------------------------------
# Attention processors (PAG, SEG, OSEG)
# SAG uses attention map -> separate path via storing maps.
# ---------------------------------------------------------------------------

class CosmosIdentityAttnProcessor:
    """PAG: replace attention output with value-only (identity attention map).
    Equivalent to using uniform/identity attention on self-attn (attn1 only).
    """

    def __call__(self, attn, hidden_states, encoder_hidden_states=None,
                 attention_mask=None, image_rotary_emb=None):
        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        value = attn.to_v(encoder_hidden_states)
        value = value.unflatten(2, (attn.heads, -1)).transpose(1, 2)
        # GQA-style key/value head match
        # For PAG: skip Q/K entirely, just propagate V averaged over identity.
        # Identity attention: each token attends only to itself => output = V.
        hidden_states = value.transpose(1, 2).flatten(2, 3).type_as(value)
        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)
        return hidden_states


class CosmosBlurredQAttnProcessor:
    """SEG: gaussian-blur query tokens spatially before SDPA. Equivalent (up to
    constants) to blurring attention logits along the query axis. sigma >> 1 --
    near-uniform attention; sigma=1e10 -> identity-like.

    Stores spatial layout (T, H, W) on attn.processor before call.
    """

    def __init__(self, sigma: float, spatial_shape):
        self.sigma = sigma
        self.spatial_shape = spatial_shape  # (T, H, W) post-patch

    @staticmethod
    def _gauss_kernel1d(sigma: float, device, dtype):
        radius = max(int(3 * sigma), 1)
        x = torch.arange(-radius, radius + 1, device=device, dtype=dtype)
        k = torch.exp(-(x ** 2) / (2 * sigma ** 2))
        return (k / k.sum()).view(1, 1, -1)

    def _blur_q(self, q: torch.Tensor) -> torch.Tensor:
        # q: [B, heads, S, dim], S = T*H*W
        if self.sigma >= 1e9:
            # identity-like attention -> replace q with its mean per sample
            return q.mean(dim=2, keepdim=True).expand_as(q)
        T, H, W = self.spatial_shape
        B, Hd, S, D = q.shape
        assert S == T * H * W, f"SEG processor S={S} != {T}*{H}*{W}"
        # operate on float32 for stability
        x = q.float().permute(0, 1, 3, 2).reshape(B * Hd * D, 1, T, H, W)
        kH = self._gauss_kernel1d(self.sigma, q.device, torch.float32).view(1, 1, 1, -1, 1)
        kW = self._gauss_kernel1d(self.sigma, q.device, torch.float32).view(1, 1, 1, 1, -1)
        rH = (kH.shape[-2] - 1) // 2
        rW = (kW.shape[-1] - 1) // 2
        x = F.pad(x, (rW, rW, rH, rH, 0, 0), mode="reflect")
        x = F.conv3d(x, kH.expand(1, 1, 1, kH.shape[-2], 1))
        x = F.conv3d(x, kW.expand(1, 1, 1, 1, kW.shape[-1]))
        x = x.reshape(B, Hd, D, T * H * W).permute(0, 1, 3, 2)
        return x.type_as(q)

    def __call__(self, attn, hidden_states, encoder_hidden_states=None,
                 attention_mask=None, image_rotary_emb=None):
        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        query = attn.to_q(hidden_states)
        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)
        query = query.unflatten(2, (attn.heads, -1)).transpose(1, 2)
        key = key.unflatten(2, (attn.heads, -1)).transpose(1, 2)
        value = value.unflatten(2, (attn.heads, -1)).transpose(1, 2)
        query = attn.norm_q(query)
        key = attn.norm_k(key)
        if image_rotary_emb is not None:
            from diffusers.models.embeddings import apply_rotary_emb
            query = apply_rotary_emb(query, image_rotary_emb, use_real=True, use_real_unbind_dim=-2)
            key = apply_rotary_emb(key, image_rotary_emb, use_real=True, use_real_unbind_dim=-2)
        # GQA repeat (key/value heads -> query heads)
        qh = query.size(3); kh = key.size(3); vh = value.size(3)
        key = key.repeat_interleave(qh // kh, dim=3)
        value = value.repeat_interleave(qh // vh, dim=3)
        # SEG: blur Q
        query = self._blur_q(query)
        out = F.scaled_dot_product_attention(query, key, value, attn_mask=attention_mask,
                                             dropout_p=0.0, is_causal=False)
        out = out.transpose(1, 2).flatten(2, 3).type_as(query)
        out = attn.to_out[0](out)
        out = attn.to_out[1](out)
        return out


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

class Cosmos2MethodsPipeline(Cosmos2TextToImagePipeline):
    """Drop-in replacement for `Cosmos2TextToImagePipeline` with `method` arg.

    Notes:
      - safety_checker is bypassed (research use).
      - num_frames is fixed to 1 (text2image).
    """

    def __init__(self, text_encoder, tokenizer, transformer, vae, scheduler, safety_checker=None):
        from diffusers.pipelines.pipeline_utils import DiffusionPipeline
        from diffusers.video_processor import VideoProcessor
        DiffusionPipeline.__init__(self)
        if safety_checker is None:
            safety_checker = _NoSafety()
        self.register_modules(
            vae=vae, text_encoder=text_encoder, tokenizer=tokenizer,
            transformer=transformer, scheduler=scheduler, safety_checker=safety_checker,
        )
        self.vae_scale_factor_temporal = 2 ** sum(self.vae.temperal_downsample) if getattr(self, "vae", None) else 4
        self.vae_scale_factor_spatial = 2 ** len(self.vae.temperal_downsample) if getattr(self, "vae", None) else 8
        self.video_processor = VideoProcessor(vae_scale_factor=self.vae_scale_factor_spatial)
        self.sigma_max = 80.0
        self.sigma_min = 0.002
        self.sigma_data = 1.0
        self.final_sigmas_type = "sigma_min"
        if self.scheduler is not None:
            self.scheduler.register_to_config(
                sigma_max=self.sigma_max, sigma_min=self.sigma_min,
                sigma_data=self.sigma_data, final_sigmas_type=self.final_sigmas_type,
            )

    def _set_attn_processor_on_layers(self, attn_attr: str, layer_indices: List[int],
                                      processor_factory: Callable):
        """Swap `attn_attr` (e.g. 'attn1') processor on selected blocks. Returns
        a restore-fn capturing originals.
        """
        originals = {}
        for idx in layer_indices:
            block = self.transformer.transformer_blocks[idx]
            attn = getattr(block, attn_attr)
            originals[idx] = attn.processor
            attn.set_processor(processor_factory(idx))

        def restore():
            for idx, proc in originals.items():
                getattr(self.transformer.transformer_blocks[idx], attn_attr).set_processor(proc)

        return restore

    def _spatial_shape(self, height: int, width: int) -> tuple:
        p_t, p_h, p_w = self.transformer.config.patch_size
        T = 1 // p_t
        H = height // self.vae_scale_factor_spatial // p_h
        W = width // self.vae_scale_factor_spatial // p_w
        return (T, H, W)

    def _resolve_layers(self, spec, num_layers: int) -> List[int]:
        if spec is None:
            return []
        if isinstance(spec, int):
            return [spec]
        if isinstance(spec, (list, tuple)):
            out = []
            for x in spec:
                if isinstance(x, int):
                    out.append(x)
                elif isinstance(x, str) and x.startswith("d") and x[1:].isdigit():
                    out.append(int(x[1:]))
                else:
                    raise ValueError(f"bad layer spec: {x}")
            return [i for i in out if 0 <= i < num_layers]
        raise ValueError(f"bad layer spec: {spec}")

    @torch.no_grad()
    def __call__(
        self,
        prompt: Union[str, List[str]] = None,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        height: int = 768,
        width: int = 1360,
        num_inference_steps: int = 35,
        guidance_scale: float = 7.0,
        num_images_per_prompt: Optional[int] = 1,
        generator=None,
        latents=None,
        prompt_embeds=None,
        negative_prompt_embeds=None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        max_sequence_length: int = 512,
        # method dispatch
        method: str = "cfg",
        # cfgpp
        cfgpp_w: float = 0.7,
        # cfg0s
        use_zero_init: bool = True,
        zero_steps: int = 0,
        # apg
        apg_eta: float = 0.0,
        apg_step_radius: float = 15.0,
        apg_momentum: float = -0.5,
        # tcfg
        tcfg_rank: int = 1,
        # sag/pag/seg/oseg
        seg_scale: float = 3.0,
        seg_blur_sigma: float = 10.0,
        seg_applied_layers: Optional[List] = None,
        oseg_scale: float = 0.5,
        pag_scale: float = 3.0,
        pag_applied_layers: Optional[List] = None,
        sag_scale: float = 0.4,
        sag_applied_layers: Optional[List] = None,
        sag_blur_sigma: float = 1.0,
        **_unused,
    ):
        # 1. Setup
        self._guidance_scale = guidance_scale
        self._current_timestep = None
        self._interrupt = False
        device = self._execution_device
        do_cfg = guidance_scale > 1.0 and method != "no_cfg"

        if isinstance(prompt, str):
            batch_size = 1
        elif isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        # 2. Encode prompt (always need cond + uncond unless no_cfg)
        prompt_embeds, negative_prompt_embeds = self.encode_prompt(
            prompt=prompt, negative_prompt=negative_prompt,
            do_classifier_free_guidance=do_cfg,
            num_images_per_prompt=num_images_per_prompt,
            prompt_embeds=prompt_embeds, negative_prompt_embeds=negative_prompt_embeds,
            device=device, max_sequence_length=max_sequence_length,
        )

        # 3. Timesteps
        sigmas_dtype = torch.float64
        sigmas = torch.linspace(0, 1, num_inference_steps, dtype=sigmas_dtype)
        timesteps, num_inference_steps = retrieve_timesteps(self.scheduler, device=device, sigmas=sigmas)
        if self.scheduler.config.get("final_sigmas_type", "zero") == "sigma_min":
            self.scheduler.sigmas[-1] = self.scheduler.sigmas[-2]

        # 4. Latents
        transformer_dtype = self.transformer.dtype
        num_channels_latents = self.transformer.config.in_channels
        latents = self.prepare_latents(
            batch_size * num_images_per_prompt, num_channels_latents,
            height, width, num_frames=1,
            dtype=torch.float32, device=device, generator=generator, latents=latents,
        )
        padding_mask = latents.new_zeros(1, 1, height, width, dtype=transformer_dtype)
        spatial_shape = self._spatial_shape(height, width)

        num_layers = len(self.transformer.transformer_blocks)
        seg_layers = self._resolve_layers(seg_applied_layers, num_layers)
        pag_layers = self._resolve_layers(pag_applied_layers, num_layers)
        sag_layers = self._resolve_layers(sag_applied_layers, num_layers)

        # APG state
        apg_prev_delta = None

        def base_forward(lat, embeds):
            current_t = self._cur_t
            c_in = self._c_in
            c_skip = self._c_skip
            c_out = self._c_out
            timestep = current_t.expand(lat.shape[0]).to(transformer_dtype)
            x_in = (lat * c_in).to(transformer_dtype)
            raw = self.transformer(
                hidden_states=x_in, timestep=timestep,
                encoder_hidden_states=embeds, padding_mask=padding_mask,
                return_dict=False,
            )[0]
            return (c_skip * lat + c_out * raw.float()).to(transformer_dtype)

        def perturbed_forward(lat, embeds, attn_attr, layers, factory):
            restore = self._set_attn_processor_on_layers(attn_attr, layers, factory)
            try:
                return base_forward(lat, embeds)
            finally:
                restore()

        self._num_timesteps = len(timesteps)
        with self.progress_bar(total=num_inference_steps) as bar:
            for i, t in enumerate(timesteps):
                self._current_timestep = t
                current_sigma = self.scheduler.sigmas[i]
                self._cur_t = current_sigma / (current_sigma + 1)
                self._c_in = 1 - self._cur_t
                self._c_skip = 1 - self._cur_t
                self._c_out = -self._cur_t

                # 1. Conditional pred
                pred_cond = base_forward(latents, prompt_embeds)

                # 2. Method-specific guidance
                cfgpp_x_next_target = None
                if not do_cfg:
                    pred = pred_cond
                else:
                    pred_uncond = base_forward(latents, negative_prompt_embeds)
                    if method == "cfg":
                        pred = pred_cond + guidance_scale * (pred_cond - pred_uncond)
                    elif method == "cfgpp":
                        sigma_next = self.scheduler.sigmas[i + 1].to(latents.device, latents.dtype) \
                            if i + 1 < len(self.scheduler.sigmas) else current_sigma.to(latents.device, latents.dtype)
                        cfgpp_x_next_target = cfgpp_x_next(
                            latents, pred_cond, pred_uncond, cfgpp_w,
                            current_sigma.to(latents.device, latents.dtype), sigma_next,
                        )
                        pred = pred_cond  # placeholder; noise_pred computed separately
                    elif method == "cfg0s":
                        pred = cfg0s_combine(pred_cond, pred_uncond, guidance_scale,
                                             i, num_inference_steps, zero_steps, use_zero_init)
                    elif method == "apg":
                        pred, apg_prev_delta = apg_combine(
                            pred_cond, pred_uncond, guidance_scale,
                            apg_eta, apg_step_radius, apg_momentum, apg_prev_delta,
                        )
                    elif method == "tcfg":
                        pred = tcfg_combine(pred_cond, pred_uncond, guidance_scale, tcfg_rank)
                    elif method == "pag":
                        pred_pag = perturbed_forward(
                            latents, prompt_embeds, "attn1", pag_layers,
                            lambda idx: CosmosIdentityAttnProcessor(),
                        )
                        # PAG combines with CFG: pred = cond + cfg*(cond-uncond) + pag*(cond-pag)
                        pred = pred_cond + guidance_scale * (pred_cond - pred_uncond) \
                               + pag_scale * (pred_cond - pred_pag)
                    elif method == "seg":
                        pred_seg = perturbed_forward(
                            latents, prompt_embeds, "attn1", seg_layers,
                            lambda idx: CosmosBlurredQAttnProcessor(seg_blur_sigma, spatial_shape),
                        )
                        pred = pred_cond + guidance_scale * (pred_cond - pred_uncond) \
                               + seg_scale * (pred_cond - pred_seg)
                    elif method == "oseg":
                        pred_seg = perturbed_forward(
                            latents, prompt_embeds, "attn1", seg_layers,
                            lambda idx: CosmosBlurredQAttnProcessor(seg_blur_sigma, spatial_shape),
                        )
                        seg_delta = pred_cond - pred_seg
                        cfg_delta = pred_cond - pred_uncond
                        # OSEG eq.(14): orthogonalise CFG against SEG direction
                        # (remove CFG component parallel to SEG), then sum.
                        _, cfg_orth = project_parallel(cfg_delta, seg_delta)
                        pred = pred_cond + guidance_scale * cfg_orth \
                               + (seg_scale + oseg_scale) * seg_delta
                    elif method == "sag":
                        # SAG-lite: blur the latent spatially and rerun cond.
                        # Faithful SAG masks high-attention regions; without an
                        # attention map (SDPA fused), we approximate by blurring
                        # all latent channels with a small Gaussian (sigma=sag_blur_sigma).
                        with torch.no_grad():
                            blurred = self._latent_blur(latents, sag_blur_sigma)
                        pred_sag = base_forward(blurred, prompt_embeds)
                        pred = pred_cond + guidance_scale * (pred_cond - pred_uncond) \
                               + sag_scale * (pred_cond - pred_sag)
                    else:
                        raise ValueError(f"unknown method: {method}")

                # 3. Convert pred (x0-style) -> noise & step
                if cfgpp_x_next_target is not None:
                    delta_sigma = (sigma_next - current_sigma).to(latents.device, latents.dtype)
                    if torch.abs(delta_sigma) < 1e-8:
                        noise_pred = (latents - pred) / current_sigma
                    else:
                        noise_pred = (cfgpp_x_next_target - latents) / delta_sigma
                else:
                    noise_pred = (latents - pred) / current_sigma
                latents = self.scheduler.step(noise_pred, t, latents, return_dict=False)[0]

                if (i + 1) % self.scheduler.order == 0 or i == len(timesteps) - 1:
                    bar.update()
                if XLA_AVAILABLE:
                    xm.mark_step()

        self._current_timestep = None

        # Decode
        latents_mean = (
            torch.tensor(self.vae.config.latents_mean)
            .view(1, self.vae.config.z_dim, 1, 1, 1)
            .to(latents.device, latents.dtype)
        )
        latents_std = 1.0 / torch.tensor(self.vae.config.latents_std).view(1, self.vae.config.z_dim, 1, 1, 1).to(
            latents.device, latents.dtype
        )
        latents = latents / latents_std / self.scheduler.config.sigma_data + latents_mean
        video = self.vae.decode(latents.to(self.vae.dtype), return_dict=False)[0]
        video = self.video_processor.postprocess_video(video, output_type=output_type)
        image = [batch[0] for batch in video]
        if isinstance(video, torch.Tensor):
            image = torch.stack(image)
        elif isinstance(video, np.ndarray):
            image = np.stack(image)

        self.maybe_free_model_hooks()
        if not return_dict:
            return (image,)
        return CosmosImagePipelineOutput(images=image)

    @staticmethod
    def _latent_blur(latents: torch.Tensor, sigma: float) -> torch.Tensor:
        if sigma <= 0:
            return latents
        radius = max(int(3 * sigma), 1)
        x = torch.arange(-radius, radius + 1, device=latents.device, dtype=torch.float32)
        k1d = torch.exp(-(x ** 2) / (2 * sigma ** 2))
        k1d = (k1d / k1d.sum())
        # latents: [B, C, T, H, W]; blur H/W only
        kH = k1d.view(1, 1, 1, -1, 1)
        kW = k1d.view(1, 1, 1, 1, -1)
        B, C, T, H, W = latents.shape
        x = latents.float().reshape(B * C, 1, T, H, W)
        x = F.pad(x, (radius, radius, radius, radius, 0, 0), mode="reflect")
        x = F.conv3d(x, kH.expand(1, 1, 1, 2 * radius + 1, 1))
        x = F.conv3d(x, kW.expand(1, 1, 1, 1, 2 * radius + 1))
        return x.reshape(B, C, T, H, W).type_as(latents)


# Skip safety checker assertion in pipeline init: provide a no-op stub.
class _NoSafety:
    def to(self, *a, **k): return self
    def check_text_safety(self, *a, **k): return True
    def check_video_safety(self, video, *a, **k): return video
