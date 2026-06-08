"""Run GenEval evaluation (Mask2Former + CLIP color classifier) on a
generated image set and write the per-run summary next to the images.

GenEval layout expected::

    image_dir/
        00000/
            metadata.jsonl
            samples/{0..3}.png
        00001/...

Usage::

    scripts/eval_geneval.py --image-dir outputs/geneval/<method>_<tag> \\
        --models-dir $GENEVAL_MODELS --mm-config $GENEVAL_MM_CONFIG
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
# GenEval scorer (evaluate_images.py + summary_scores.py) lives in the external
# GenEval benchmark repo. Override with $GENEVAL_ROOT.
GENEVAL_ROOT = Path(os.environ.get("GENEVAL_ROOT", str(Path.home() / "geneval-bench")))


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--image-dir", required=True,
                   help="Folder with {NNNNN}/samples/{i}.png + metadata.jsonl")
    p.add_argument("--models-dir", required=True,
                   help="Dir with mask2former_*.pth checkpoint")
    p.add_argument("--mm-config", required=True,
                   help="Path to mmdet py config for Mask2Former")
    p.add_argument("--out-dir", default=None,
                   help="Where to place results.jsonl + summary.txt (default: --image-dir)")
    return p.parse_args()


def run(cmd, **kw):
    print(f"$ {' '.join(str(c) for c in cmd)}", flush=True)
    subprocess.run([str(c) for c in cmd], check=True, **kw)


def main():
    args = parse_args()
    image_dir = Path(args.image_dir).resolve()
    out_dir = Path(args.out_dir).resolve() if args.out_dir else image_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    if not Path(args.mm_config).is_file():
        sys.exit(f"mmdet config missing: {args.mm_config}")

    results = out_dir / "results.jsonl"
    summary = out_dir / "summary.txt"

    run([sys.executable, GENEVAL_ROOT / "evaluation" / "evaluate_images.py",
         image_dir, "--outfile", results,
         "--model-path", args.models_dir, "--model-config", args.mm_config])

    with open(summary, "w") as f:
        subprocess.run([sys.executable,
                        GENEVAL_ROOT / "evaluation" / "summary_scores.py",
                        str(results)], stdout=f, check=True)
    print(f"[geneval eval] {summary}")


if __name__ == "__main__":
    main()
