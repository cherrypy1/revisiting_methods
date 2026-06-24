"""Small shared helpers for benchmark adapters (subprocess, symlinks, env paths)."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path


def env_path(name: str, default) -> Path:
    return Path(os.environ.get(name, str(default)))


def run(cmd, cwd=None, env=None) -> None:
    cmd = [str(c) for c in cmd]
    print("$ " + " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, cwd=str(cwd) if cwd else None, env=env)


def link_or_copy(src, dst) -> None:
    """Symlink src→dst (cheap bridging); fall back to copy across filesystems."""
    src = Path(src).resolve()
    dst = Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    try:
        os.symlink(src, dst)
    except OSError:
        import shutil
        shutil.copy2(src, dst)


def to_webp(src, dst) -> None:
    """Convert an image to webp (OneIG's expected format). Lazy PIL import."""
    from PIL import Image
    dst = Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(src) as im:
        im.convert("RGB").save(dst, format="webp", quality=95)
