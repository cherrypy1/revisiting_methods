"""Create a random balanced OneIG CSV subset."""

from __future__ import annotations

import argparse
import collections
import csv
import random
from pathlib import Path


DEFAULT_CATEGORIES = [
    "Anime_Stylization",
    "Portrait",
    "General_Object",
    "Text_Rendering",
    "Knowledge_Reasoning",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-csv", required=True)
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--per-category", type=int, default=25)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--categories", nargs="+", default=DEFAULT_CATEGORIES)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    src = Path(args.input_csv)
    dst = Path(args.output_csv)
    dst.parent.mkdir(parents=True, exist_ok=True)

    with src.open(newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        fieldnames = reader.fieldnames

    if not fieldnames or "category" not in fieldnames:
        raise SystemExit(f"OneIG csv must contain a category column: {src}")

    by_category: dict[str, list[dict[str, str]]] = collections.defaultdict(list)
    wanted = set(args.categories)
    for row in rows:
        category = row.get("category", "")
        if category in wanted:
            by_category[category].append(row)

    rng = random.Random(args.seed)
    chosen = []
    for category in args.categories:
        items = list(by_category.get(category, []))
        if len(items) < args.per_category:
            raise SystemExit(f"not enough OneIG prompts for {category}: {len(items)} < {args.per_category}")
        rng.shuffle(items)
        chosen.extend(items[: args.per_category])

    rng.shuffle(chosen)
    with dst.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(chosen)

    print(f"[oneig prompts] {len(chosen)} rows -> {dst}")
    for category in args.categories:
        print(f"  {category}: {args.per_category}")


if __name__ == "__main__":
    main()
