"""Sync custom diffusers pipelines from this repo to remote ``~/diffusers``.

Copies ``pipelines/sd35/pipeline_stable_diffusion_3_*.py`` into
``~/diffusers/src/diffusers/pipelines/stable_diffusion_3/`` (and the Cosmos2
analogue). After sync, configs importing
``diffusers.pipelines.stable_diffusion_3.pipeline_stable_diffusion_3_X``
pick up the latest local edits.

Usage:
    python scripts/sync_pipelines.py            # both sd35 and cosmos2
    python scripts/sync_pipelines.py --only sd35
    python scripts/sync_pipelines.py --host hse-hpc --dry-run
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Local source dir → remote dest dir (relative to remote $HOME).
TARGETS = {
    "sd35": (
        PROJECT_ROOT / "pipelines" / "sd35",
        "~/diffusers/src/diffusers/pipelines/stable_diffusion_3/",
        "pipeline_stable_diffusion_3*.py",
    ),
    "cosmos2": (
        PROJECT_ROOT / "pipelines" / "cosmos2",
        "~/diffusers/src/diffusers/pipelines/",  # cosmos2 lives one level up
        "pipeline_cosmos2*.py",
    ),
}


def sync(key, host, dry_run):
    src_dir, remote_dir, pattern = TARGETS[key]
    files = sorted(src_dir.glob(pattern))
    if not files:
        print(f"[{key}] no files match {src_dir}/{pattern}")
        return
    print(f"[{key}] {len(files)} files -> {host}:{remote_dir}")
    for f in files:
        print(f"  {f.name}")
    if dry_run:
        return
    cmd = ["rsync", "-az", "--info=progress2",
           *[str(f) for f in files],
           f"{host}:{remote_dir}"]
    subprocess.run(cmd, check=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--host", default="hse-hpc")
    p.add_argument("--only", choices=list(TARGETS), help="restrict to one target")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    keys = [args.only] if args.only else list(TARGETS)
    for k in keys:
        sync(k, args.host, args.dry_run)


if __name__ == "__main__":
    main()
