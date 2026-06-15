"""Build comparison grids per method for cosmos2 HP sweep.

Per method, one PNG: rows = trials, cols = prompts. Each cell downsized,
prompt header above columns, trial name on left of rows.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


def load_font(size: int):
    for path in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/TTF/DejaVuSans.ttf",
    ):
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            continue
    return ImageFont.load_default()


def text_size(draw, text, font):
    bbox = draw.textbbox((0, 0), text, font=font)
    return bbox[2] - bbox[0], bbox[3] - bbox[1]


def wrap(draw, text, font, max_w):
    words = text.split()
    lines, cur = [], ""
    for w in words:
        cand = f"{cur} {w}".strip()
        if text_size(draw, cand, font)[0] <= max_w:
            cur = cand
        else:
            if cur:
                lines.append(cur)
            cur = w
    if cur:
        lines.append(cur)
    return lines


def build_grid(method_dir: Path, out_path: Path, cell: int = 256, max_prompts: int = 12):
    trials = sorted(p for p in method_dir.iterdir() if p.is_dir())
    if not trials:
        print(f"skip {method_dir.name}: no trials")
        return
    # Collect prompts from first trial (all trials share same chosen prompts)
    sample_dir = trials[0]
    sample_idxs = sorted(p.name for p in sample_dir.iterdir() if p.is_dir())
    sample_idxs = sample_idxs[:max_prompts]
    prompts = []
    for idx in sample_idxs:
        meta_path = sample_dir / idx / "metadata.jsonl"
        if meta_path.exists():
            with open(meta_path) as fp:
                meta = json.loads(fp.readline())
            prompts.append(meta.get("prompt", idx))
        else:
            prompts.append(idx)

    n_rows, n_cols = len(trials), len(sample_idxs)
    label_w = 360
    header_h = 110
    font = load_font(13)
    small = load_font(11)

    W = label_w + n_cols * cell
    H = header_h + n_rows * cell
    canvas = Image.new("RGB", (W, H), "white")
    draw = ImageDraw.Draw(canvas)

    # Headers
    for c, prompt in enumerate(prompts):
        x = label_w + c * cell
        lines = wrap(draw, prompt, small, cell - 6)[:6]
        y = 4
        for line in lines:
            draw.text((x + 3, y), line, fill="black", font=small)
            y += text_size(draw, line, small)[1] + 1

    for r, trial in enumerate(trials):
        y = header_h + r * cell
        lines = wrap(draw, trial.name, font, label_w - 8)
        ty = y + 4
        for line in lines:
            draw.text((4, ty), line, fill="black", font=font)
            ty += text_size(draw, line, font)[1] + 1
        for c, idx in enumerate(sample_idxs):
            img_path = trial / idx / "samples" / "00000.png"
            x = label_w + c * cell
            if not img_path.exists():
                draw.rectangle([x, y, x + cell, y + cell], outline="red")
                draw.text((x + 4, y + 4), "MISS", fill="red", font=small)
                continue
            im = Image.open(img_path).convert("RGB").resize((cell, cell), Image.LANCZOS)
            canvas.paste(im, (x, y))
            im.close()
            draw.rectangle([x, y, x + cell - 1, y + cell - 1], outline="black")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path, optimize=True)
    print(f"{method_dir.name}: {n_rows}x{n_cols} -> {out_path} ({out_path.stat().st_size//1024}KB)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(Path.home() / "geneval" / "outputs" / "cosmos2" / "hp"))
    ap.add_argument("--out", default=str(Path.home() / "geneval" / "outputs" / "cosmos2" / "hp_grids"))
    ap.add_argument("--cell", type=int, default=256)
    args = ap.parse_args()
    root, out = Path(args.root), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for m in sorted(p for p in root.iterdir() if p.is_dir()):
        build_grid(m, out / f"{m.name}.png", cell=args.cell)


if __name__ == "__main__":
    main()
