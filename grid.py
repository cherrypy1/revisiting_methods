from pathlib import Path
import re
from types import ModuleType

from PIL import Image, ImageDraw, ImageFont

from inference import load_final_configs, load_params as load_method_params


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


def method_baseline_params(method):
    method_name = method_output_name(method)
    baseline_params = {
        "pag": {"pag_scale": 0.0},
        "sag": {"sag_scale": 0.0},
        "seg": {"seg_scale": 0.0},
        "oseg": {"seg_scale": 0.0, "oseg_scale": 0.0},
        "apg": {"use_apg": False},
        "cfgpp": {"use_cfgpp": False},
        "cfg0s": {"use_cfg_zero_star": False, "use_zero_init": False},
        "tcfg": {"use_tcfg": False},
        "cfg": {},
    }
    return dict(baseline_params.get(method_name, {}))


def sweep_has_baseline_column(sweep, baseline_params):
    if not baseline_params:
        return True

    row_name, _ = sweep["row"]
    col_name, col_values = sweep["col"]
    fixed = sweep.get("fixed", {})

    if row_name in baseline_params:
        return False

    for col_value in col_values:
        params = dict(fixed)
        params[col_name] = col_value
        if all(params.get(key) == value for key, value in baseline_params.items()):
            return True

    return False


def normalize_configs(configs):
    result = []
    for i, config in enumerate(configs, start=1):
        if isinstance(config, tuple) and len(config) == 2:
            label, params = config
        elif isinstance(config, dict):
            label = config.get("label", config.get("name"))
            if "params" in config:
                params = config["params"]
            else:
                params = {
                    key: value
                    for key, value in config.items()
                    if key not in {"label", "name"}
                }
        else:
            raise TypeError("Each config must be a dict or (label, params) tuple.")

        result.append((label or f"config {i}", dict(params)))
    return result


def config_label(label, params):
    text = display_params_str(params)
    if text:
        return f"{label}\n{text}"
    return str(label)


def add_shared_params(target, configs, keys):
    for key in keys:
        values = [params[key] for _, params in configs if key in params]
        if not values or len(values) != len(configs):
            continue
        first_value = values[0]
        if all(value == first_value for value in values[1:]):
            target.setdefault(key, first_value)


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


def fit_wrapped_text(draw, text, max_width, max_size, min_size, bold, target_lines):
    for size in range(max_size, min_size - 1, -1):
        font = get_font(size, bold=bold)
        lines = wrap_text(draw, text, font, max_width)
        if len(lines) <= target_lines:
            return font, lines

    font = get_font(min_size, bold=bold)
    return font, wrap_text(draw, text, font, max_width)


def resize_image(image, size):
    image = image.convert("RGB")
    image.thumbnail((size, size), Image.Resampling.LANCZOS)

    tile = Image.new("RGB", (size, size), "white")
    x = (size - image.width) // 2
    y = (size - image.height) // 2
    tile.paste(image, (x, y))
    return tile


def draw_grid(
    images,
    row_labels,
    col_labels,
    title,
    subtitle,
    tile_size=384,
    left=218,
    header=50,
    row_label_lines=6,
    col_label_lines=2,
):
    rows = len(images)
    cols = len(images[0])

    margin = 14
    gap = 6
    min_top = 108

    width = margin * 2 + left + cols * tile_size + (cols - 1) * gap
    measure = ImageDraw.Draw(Image.new("RGB", (1, 1)))
    title_font, title_lines = fit_wrapped_text(
        measure,
        title,
        width - 2 * margin,
        max_size=21,
        min_size=13,
        bold=True,
        target_lines=4,
    )
    text_font, subtitle_lines = fit_wrapped_text(
        measure,
        subtitle,
        width - 2 * margin,
        max_size=13,
        min_size=11,
        bold=False,
        target_lines=3,
    )
    label_font = get_font(16, bold=True)
    title_line_height = title_font.size + 5 if hasattr(title_font, "size") else 20
    subtitle_line_height = text_font.size + 4 if hasattr(text_font, "size") else 16
    line_height = 21

    heading_height = (
        len(title_lines) * title_line_height
        + len(subtitle_lines) * subtitle_line_height
    )
    top = max(min_top, heading_height + header + 8)
    height = margin * 2 + top + rows * tile_size + (rows - 1) * gap

    grid = Image.new("RGB", (width, height), "#f8fafc")
    draw = ImageDraw.Draw(grid)

    y = margin
    for line in title_lines:
        draw.text((margin, y), line, fill="#0f172a", font=title_font)
        y += title_line_height
    for line in subtitle_lines:
        draw.text((margin, y), line, fill="#475569", font=text_font)
        y += subtitle_line_height

    x0 = margin + left
    y0 = margin + top - header

    for col, label in enumerate(col_labels):
        x = x0 + col * (tile_size + gap)
        draw.rounded_rectangle(
            [x, y0, x + tile_size, y0 + header - gap], radius=5, fill="#e8eef6"
        )
        label_lines = wrap_text(draw, label, label_font, tile_size - 20)[
            :col_label_lines
        ]
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
        label_lines = wrap_text(draw, label, label_font, left - 24)[:row_label_lines]
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
    import torch

    device = getattr(pipe, "_execution_device", None)
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.Generator(device=device).manual_seed(seed)


def run_sweep(pipe, prompt, method, sweep, base_params, try_id, seed, tile_size):
    row_name, row_values = sweep["row"]
    col_name, col_values = sweep["col"]
    fixed = sweep.get("fixed", {})
    baseline_params = method_baseline_params(method)
    add_baseline = not sweep_has_baseline_column(sweep, baseline_params)

    images = []
    for row_value in row_values:
        image_row = []

        if add_baseline:
            params = dict(base_params)
            params.update(fixed)
            params[row_name] = row_value
            params.update(baseline_params)

            image = pipe(
                prompt=prompt,
                generator=make_generator(pipe, seed),
                **params,
            ).images[0]
            image_row.append(image)

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
    if add_baseline:
        col_labels = ["CFG baseline"] + col_labels
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
    finals=False,
):
    if finals:
        return [
            run_method_configs_grid(
                pipe=pipe,
                method=method,
                prompts=prompts,
                try_id=try_id,
                seed=seed,
                tile_size=tile_size,
                method_name=method_name,
                grid_name="finals",
            )
        ]

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


def run_method_configs_grid(
    pipe,
    method,
    prompts,
    configs=None,
    try_id="2705",
    seed=7,
    tile_size=384,
    method_name=None,
    grid_name="selected_configs",
    baseline_params=None,
    baseline_label="CFG baseline",
):
    base_params, _ = load_method_params(method)
    output_method = method_output_name(method, method_name)
    if configs is None:
        configs = load_final_configs(method)
    selected_configs = normalize_configs(configs)
    if not selected_configs:
        raise ValueError(f"No FINALS found for method: {output_method}")

    if baseline_params is None:
        baseline_params = method_baseline_params(output_method)
    else:
        baseline_params = dict(baseline_params)

    add_shared_params(
        baseline_params,
        selected_configs,
        keys=("guidance_scale", "num_inference_steps", "height", "width"),
    )

    all_configs = [(baseline_label, baseline_params)] + selected_configs
    images = [[None for _ in all_configs] for _ in prompts]

    # Generate by columns: all prompts for one fixed config, then the next config.
    for col, (label, config_params) in enumerate(all_configs):
        params = dict(base_params)
        params.update(config_params)

        for row, prompt in enumerate(prompts):
            image = pipe(
                prompt=prompt,
                generator=make_generator(pipe, seed),
                **params,
            ).images[0]
            images[row][col] = image

        print(f"done: {output_method} / {label}")

    row_labels = [f"{i}. {prompt}" for i, prompt in enumerate(prompts, start=1)]
    col_labels = [config_label(label, params) for label, params in all_configs]
    title = f"{output_method}: selected configurations"
    subtitle = f"seed={seed}; columns are generated one config at a time"

    grid = draw_grid(
        images,
        row_labels,
        col_labels,
        title,
        subtitle,
        tile_size=tile_size,
        left=360,
        header=92,
        row_label_lines=10,
        col_label_lines=4,
    )

    out_dir = Path("outputs") / output_method / str(try_id)
    out_dir.mkdir(parents=True, exist_ok=True)

    out_path = out_dir / f"{grid_name}.jpg"
    grid.save(out_path, "JPEG", quality=94, optimize=True, progressive=True)
    print(f"saved: {out_path}")
    return out_path
