"""Compose a visual grid from a sweep directory (or any set of sub-folders).

Layout assumed:
    root/
        {col_name_1}/{idx}.png
        {col_name_2}/{idx}.png
        ...

Output: one PNG where rows = prompts (matched by filename), cols = sub-folders,
each cell labelled with its column name. Useful for eyeballing sweep results
produced by ``scripts/sweep.py``.

Usage:
    scripts/make_grid.py outputs/sweep/cfgpp_scale --out cfgpp_scale_grid.png
    scripts/make_grid.py outputs/sweep/cfgpp_scale --cell 256 --rows 8
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("root", help="Directory containing one subfolder per column")
    p.add_argument("--out", default=None, help="Output path (default: {root}/_grid.png)")
    p.add_argument("--cell", type=int, default=256, help="Per-cell thumbnail side (px)")
    p.add_argument("--rows", type=int, default=None, help="Limit number of prompt rows")
    p.add_argument("--cols", nargs="+", default=None,
                   help="Explicit column order (subfolder names). Default: sorted.")
    p.add_argument("--header", type=int, default=28, help="Header strip height (px)")
    p.add_argument("--label", type=int, default=180,
                   help="Left label strip width (px) for prompt preview, 0 to disable")
    return p.parse_args()


def list_cols(root, wanted):
    subs = sorted(p.name for p in Path(root).iterdir() if p.is_dir())
    if wanted:
        missing = [c for c in wanted if c not in subs]
        if missing:
            raise SystemExit(f"missing column dirs: {missing} (have {subs})")
        return wanted
    return subs


def collect_rows(col_dirs, rows_cap):
    keys = None
    for d in col_dirs:
        names = sorted(f.stem for f in d.iterdir() if f.suffix == ".png")
        names_set = set(names)
        keys = names_set if keys is None else keys & names_set
    keys = sorted(keys)
    if rows_cap:
        keys = keys[:rows_cap]
    return keys


def load_font(size):
    for path in ["/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
                 "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf"]:
        if os.path.exists(path):
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def read_prompt(col_dir, stem):
    p = col_dir / f"{stem}.txt"
    if p.exists():
        return p.read_text().strip()
    return stem


def main():
    args = parse_args()
    root = Path(args.root)
    col_names = list_cols(root, args.cols)
    col_dirs = [root / c for c in col_names]

    keys = collect_rows(col_dirs, args.rows)
    if not keys:
        raise SystemExit("No matching image stems across columns.")

    cell = args.cell
    header = args.header
    label = args.label
    W = label + cell * len(col_names)
    H = header + cell * len(keys)
    canvas = Image.new("RGB", (W, H), "white")
    draw = ImageDraw.Draw(canvas)
    font = load_font(14)

    for ci, name in enumerate(col_names):
        x = label + ci * cell
        draw.rectangle([x, 0, x + cell, header], fill="#f0f0f0")
        draw.text((x + 6, 6), name, fill="black", font=font)

    for ri, stem in enumerate(keys):
        y = header + ri * cell
        if label:
            prompt = read_prompt(col_dirs[0], stem)
            draw.rectangle([0, y, label, y + cell], fill="#fafafa")
            wrapped = prompt[:220]
            draw.text((4, y + 4), f"{stem}\n{wrapped}", fill="black", font=font)
        for ci, d in enumerate(col_dirs):
            img = Image.open(d / f"{stem}.png").convert("RGB")
            img.thumbnail((cell, cell), Image.LANCZOS)
            ox = label + ci * cell + (cell - img.width) // 2
            oy = y + (cell - img.height) // 2
            canvas.paste(img, (ox, oy))

    out = Path(args.out) if args.out else root / "_grid.png"
    canvas.save(out)
    print(f"grid → {out} ({W}x{H})")


if __name__ == "__main__":
    main()
