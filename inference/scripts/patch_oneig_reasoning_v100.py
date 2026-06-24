"""Patch OneIG reasoning/LLM2Vec remote code so it can run without FlashAttention2.

OneIG reasoning uses LLM2CLIP, whose LLM2Vec remote module raises unless the
attention implementation is FlashAttention2. V100 nodes do not support the
usual FlashAttention2 wheels, so this patch turns that hard failure into an
eager-attention fallback.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path


LLM2CLIP_LLM = "microsoft/LLM2CLIP-Llama-3-8B-Instruct-CC-Finetuned"
ERROR_TEXT = (
    "The current implementation of LlamaBiModel only supports flash attention 2 "
    "for attention implementation"
)


def candidate_roots() -> list[Path]:
    roots = []
    for key in ("HF_HOME", "HUGGINGFACE_HUB_CACHE", "TRANSFORMERS_CACHE"):
        value = os.environ.get(key)
        if value:
            roots.append(Path(value).expanduser())

    home = Path.home()
    roots.extend([
        home / ".cache" / "huggingface",
        Path("/tmp") / os.environ.get("USER", "") / "hf_oneig_reasoning",
    ])
    return list(dict.fromkeys(roots))


def prepare_remote_code() -> None:
    try:
        from transformers import AutoConfig

        AutoConfig.from_pretrained(LLM2CLIP_LLM, trust_remote_code=True)
    except Exception as exc:
        print(f"warning: could not prefetch LLM2Vec remote code: {exc}")


def force_remote_model_download() -> None:
    try:
        import torch
        from transformers import AutoConfig, AutoModel

        config = AutoConfig.from_pretrained(LLM2CLIP_LLM, trust_remote_code=True)
        AutoModel.from_pretrained(
            LLM2CLIP_LLM,
            torch_dtype=torch.bfloat16,
            config=config,
            trust_remote_code=True,
        )
    except Exception as exc:
        if ERROR_TEXT in str(exc):
            print("expected FlashAttention2 guard hit while preparing remote code")
        else:
            print(f"warning: remote model preflight failed: {exc}")


def patch_guard(path: Path) -> bool:
    text = path.read_text()
    if ERROR_TEXT not in text:
        return False

    lines = text.splitlines()
    error_at = None
    for i, line in enumerate(lines):
        if "current implementation of LlamaBiModel" in line:
            error_at = i
            break
    if error_at is None:
        raise RuntimeError(f"found FlashAttention2 guard but could not patch it: {path}")

    raise_at = None
    for i in range(error_at, max(-1, error_at - 12), -1):
        if "raise ValueError" in lines[i]:
            raise_at = i
            break
    if raise_at is None:
        raise RuntimeError(f"found error text but no nearby raise ValueError: {path}")

    indent = lines[raise_at][: len(lines[raise_at]) - len(lines[raise_at].lstrip())]
    end_at = raise_at
    depth = 0
    seen_open = False
    while end_at < len(lines):
        for ch in lines[end_at]:
            if ch == "(":
                depth += 1
                seen_open = True
            elif ch == ")":
                depth -= 1
        if seen_open and depth <= 0:
            break
        end_at += 1

    replacement = [f'{indent}config._attn_implementation = "eager"']
    lines = lines[:raise_at] + replacement + lines[end_at + 1 :]
    path.write_text("\n".join(lines) + "\n")
    return True


def patch_cached_modules() -> int:
    patched = 0
    seen = set()
    for root in candidate_roots():
        if not root.exists():
            continue
        for path in root.rglob("modeling_llama_encoder.py"):
            if path in seen:
                continue
            seen.add(path)
            try:
                if patch_guard(path):
                    print(f"patched: {path}")
                    patched += 1
            except Exception as exc:
                print(f"warning: failed to patch {path}: {exc}")
    return patched


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepare-code", action="store_true")
    args = parser.parse_args()

    if args.prepare_code:
        prepare_remote_code()

    patched = patch_cached_modules()
    if patched == 0 and args.prepare_code:
        force_remote_model_download()
        patched = patch_cached_modules()
    if patched == 0:
        print("warning: no cached modeling_llama_encoder.py was patched")


if __name__ == "__main__":
    main()
