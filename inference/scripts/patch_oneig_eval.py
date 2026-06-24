"""Patch external OneIG-Benchmark eval for V100/cluster runs.

OneIG's Qwen2.5-VL helper may force ``attn_implementation="flash_attention_2"``.
The cluster V100 nodes do not have flash-attn, so evaluation fails before scoring.
This patch switches that explicit request to PyTorch SDPA.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--oneig-root", default=os.environ.get("ONEIG_ROOT", str(Path.home() / "OneIG-Benchmark")))
    args = p.parse_args()

    target = Path(args.oneig_root) / "scripts" / "utils" / "inference.py"
    if not target.is_file():
        raise SystemExit(f"OneIG inference helper not found: {target}")

    text = target.read_text()
    patched = text.replace('"flash_attention_2"', '"sdpa"').replace("'flash_attention_2'", "'sdpa'")
    if patched == text:
        if "sdpa" in text:
            print(f"already patched: {target}")
            return
        raise SystemExit(f"flash_attention_2 literal not found in {target}")

    target.write_text(patched)
    print(f"patched flash_attention_2 -> sdpa in {target}")


if __name__ == "__main__":
    main()
