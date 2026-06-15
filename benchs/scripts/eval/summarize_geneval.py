"""Walk outputs/geneval/*_<tag>/results.jsonl, print upstream summary per method.

Replaces ``summarize_geneval.sh``. Usage:
    python scripts/eval/summarize_geneval.py <run_tag>
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
# summary_scores.py ships with the external GenEval benchmark repo.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
GENEVAL_ROOT = Path(os.environ.get("GENEVAL_ROOT", str(PROJECT_ROOT / "geneval-bench")))
SUMMARY = GENEVAL_ROOT / "evaluation" / "summary_scores.py"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("tag", help="run-tag suffix (e.g. final22042026)")
    args = p.parse_args()

    runs = sorted((PROJECT_ROOT / "outputs" / "geneval").glob(f"*_{args.tag}"))
    if not runs:
        print(f"No runs match outputs/geneval/*_{args.tag}", file=sys.stderr)
        sys.exit(1)

    for d in runs:
        f = d / "results.jsonl"
        if not f.exists():
            continue
        print(f"########## {d.name} ##########")
        subprocess.run([sys.executable, str(SUMMARY), str(f)], check=False)
        print()


if __name__ == "__main__":
    main()
