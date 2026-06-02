from pathlib import Path
import re
from types import ModuleType

import torch
from PIL import Image, ImageDraw, ImageFont

from inference import load_params as load_method_params


HIDDEN_DISPLAY_PARAMS = {
    "height",
    "width",
    "image_shape",
    "image_size",
    "num_images_per_prompt",
    "output_type",
    "return_dict",
}

FONT_CANDIDATES = {
    False: [
        "/usr/share/fonts/truetype/ubuntu/Ubuntu-R.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
        "DejaVuSans.ttf",
    ],
    True: [
        "/usr/share/fonts/truetype/ubuntu/Ubuntu-M.ttf",
        "/usr/share/fonts/truetype/ubuntu/Ubuntu-B.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf",
        "DejaVuSans-Bold.ttf",
    ],
}


def method_output_name(method, method_name=None):
    if method_name is not None:
        return method_name
    if isinstance(method, str):
        return method.rsplit(".", 1)[-1]
    if isinstance(method, ModuleType):
        return method.__name__.rsplit(".", 1)[-1]
    if callable(method):
        module_name = getattr(method, "__module__", "")
        if module_name.startswith("_flux2_params_"):
            return module_name.removeprefix("_flux2_params_")
        if module_name.startswith("params."):
            return module_name.rsplit(".", 1)[-1]
        name = method.__name__
        for suffix in ("_params", "_config"):
            if name.endswith(suffix):
                name = name[: -len(suffix)]
        return name
    return "custom"


def short_name(prompt, limit=12):
    name = prompt[:limit].strip().lower()
    name = re.sub(r"[^a-z0-9а-яё]+", "_", name, flags=re.IGNORECASE).strip("_")
    return name or "prompt"


def value_str(value):
    if value is None:
        return "all"
    if isinstance(value, float) and value > 100000:
        return "inf"
    if isinstance(value, (list, tuple)):
        return "+".join(str(x) for x in value)
    return str(value)


def params_str(params):
    return ", ".join(f"{key}={value_str(value)}" for key, value in params.items())


def display_params_str(params):
    visible_params = {
        key: value for key, value in params.items() if key not in HIDDEN_DISPLAY_PARAMS
    }
    return params_str(visible_params)


def soft_wrap_identifier(name, limit=20):
    name = str(name)
    if len(name) <= limit:
        return name

    if "_" not in name:
        return "\n".join(name[i : i + limit] for i in range(0, len(name), limit))

    lines = []
    current = ""
    for part in name.split("_"):
        token = part if not current else f"_{part}"
        if current and len(current) + len(token) > limit:
            lines.append(f"{current}_")
            current = part
        else:
            current = f"{current}{token}"

    if current:
        lines.append(current)
    return "\n".join(lines)


def param_label(name, value, name_limit=20):
    return f"{soft_wrap_identifier(name, name_limit)} = {value_str(value)}"


def get_font(size, bold=False):
    for name in FONT_CANDIDATES[bold]:
        try:
            return ImageFont.truetype(name, size=size)
        except OSError:
            continue
    return ImageFont.load_default()


def text_width(draw, text, font):
    box = draw.textbbox((0, 0), text, font=font)
    return box[2] - box[0]


def split_long_word(draw, word, font, max_width):
    parts = []
    part = ""

    for char in word:
        test_part = part + char
        if part and text_width(draw, test_part, font) > max_width:
            parts.append(part)
            part = char
        else:
            part = test_part

    if part:
        parts.append(part)
    return parts or [word]


def wrap_text(draw, text, font, max_width):
    lines = []

    for paragraph in str(text).splitlines():
        words = paragraph.split()
        line = ""

        for word in words:
            if text_width(draw, word, font) > max_width:
                if line:
                    lines.append(line)
                    line = ""

                word_parts = split_long_word(draw, word, font, max_width)
                lines.extend(word_parts[:-1])
                line = word_parts[-1]
                continue

            test_line = f"{line} {word}".strip()
            if text_width(draw, test_line, font) <= max_width:
                line = test_line
            else:
                if line:
                    lines.append(line)
                line = word

        if line:
            lines.append(line)
    return lines or [str(text)]


def resize_image(image, size):
    image = image.convert("RGB")
    image.thumbnail((size, size), Image.Resampling.LANCZOS)

    tile = Image.new("RGB", (size, size), "white")
    x = (size - image.width) // 2
    y = (size - image.height) // 2
    tile.paste(image, (x, y))
    return tile


def draw_grid(images, row_labels, col_labels, title, subtitle, tile_size=384):
    rows = len(images)
    cols = len(images[0])

    margin = 14
    gap = 6
    left = 218
    min_top = 108
    header = 50

    width = margin * 2 + left + cols * tile_size + (cols - 1) * gap
    title_font = get_font(25, bold=True)
    text_font = get_font(15)
    label_font = get_font(16, bold=True)
    line_height = 21

    measure = ImageDraw.Draw(Image.new("RGB", (1, 1)))
    title_lines = wrap_text(measure, title, title_font, width - 2 * margin)[:2]
    subtitle_lines = wrap_text(measure, subtitle, text_font, width - 2 * margin)[:2]
    heading_height = len(title_lines) * 30 + len(subtitle_lines) * 19
    top = max(min_top, heading_height + header + 8)
    height = margin * 2 + top + rows * tile_size + (rows - 1) * gap

    grid = Image.new("RGB", (width, height), "#f8fafc")
    draw = ImageDraw.Draw(grid)

    y = margin
    for line in title_lines:
        draw.text((margin, y), line, fill="#0f172a", font=title_font)
        y += 30
    for line in subtitle_lines:
        draw.text((margin, y), line, fill="#475569", font=text_font)
        y += 19

    x0 = margin + left
    y0 = margin + top - header

    for col, label in enumerate(col_labels):
        x = x0 + col * (tile_size + gap)
        draw.rounded_rectangle(
            [x, y0, x + tile_size, y0 + header - gap], radius=5, fill="#e8eef6"
        )
        label_lines = wrap_text(draw, label, label_font, tile_size - 20)[:2]
        for i, line in enumerate(label_lines):
            draw.text(
                (x + 10, y0 + 10 + i * line_height),
                line,
                fill="#0f172a",
                font=label_font,
            )

    for row, label in enumerate(row_labels):
        y = margin + top + row * (tile_size + gap)
        draw.rounded_rectangle(
            [margin, y, margin + left - gap, y + tile_size], radius=5, fill="#e8eef6"
        )
        label_lines = wrap_text(draw, label, label_font, left - 24)[:6]
        for i, line in enumerate(label_lines):
            draw.text(
                (margin + 10, y + 12 + i * line_height),
                line,
                fill="#0f172a",
                font=label_font,
            )

        for col in range(cols):
            x = x0 + col * (tile_size + gap)
            tile = resize_image(images[row][col], tile_size)
            grid.paste(tile, (x, y))
            draw.rectangle(
                [x, y, x + tile_size - 1, y + tile_size - 1], outline="#cbd5e1"
            )

    return grid


def make_generator(pipe, seed):
    device = getattr(pipe, "_execution_device", None)
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.Generator(device=device).manual_seed(seed)


def run_sweep(pipe, prompt, method, sweep, base_params, try_id, seed, tile_size):
    row_name, row_values = sweep["row"]
    col_name, col_values = sweep["col"]
    fixed = sweep.get("fixed", {})

    images = []
    for row_value in row_values:
        image_row = []
        for col_value in col_values:
            params = dict(base_params)
            params.update(fixed)
            params[row_name] = row_value
            params[col_name] = col_value

            image = pipe(
                prompt=prompt,
                generator=make_generator(pipe, seed),
                **params,
            ).images[0]
            image_row.append(image)
        images.append(image_row)

    row_labels = [param_label(row_name, x, name_limit=16) for x in row_values]
    col_labels = [param_label(col_name, x, name_limit=24) for x in col_values]
    subtitle_parts = [f"seed={seed}"]
    fixed_text = display_params_str(fixed)
    if fixed_text:
        subtitle_parts.append(f"fixed: {fixed_text}")
    subtitle = "; ".join(subtitle_parts)

    grid = draw_grid(images, row_labels, col_labels, prompt, subtitle, tile_size)

    out_dir = Path("outputs") / method / str(try_id)
    out_dir.mkdir(parents=True, exist_ok=True)

    prompt_part = short_name(prompt)
    out_path = out_dir / f"{prompt_part}__{sweep['name']}.jpg"
    grid.save(out_path, "JPEG", quality=94, optimize=True, progressive=True)
    return out_path


def run_method_grid(
    pipe,
    method,
    prompts,
    try_id="2705",
    seed=7,
    tile_size=384,
    method_name=None,
):
    base_params, sweeps = load_method_params(method)
    output_method = method_output_name(method, method_name)
    saved_paths = []

    for prompt in prompts:
        for sweep in sweeps:
            path = run_sweep(
                pipe=pipe,
                prompt=prompt,
                method=output_method,
                sweep=sweep,
                base_params=base_params,
                try_id=try_id,
                seed=seed,
                tile_size=tile_size,
            )
            print(f"saved: {path}")
            saved_paths.append(path)

    return saved_paths
