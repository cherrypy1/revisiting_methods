"""Create a random balanced DPG-Bench subset.

The subset consists of a filtered dpg_bench.csv and a prompts/ directory whose
txt files match the selected item_id values.
"""

from __future__ import annotations

import argparse
import collections
import csv
import random
import shutil
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-csv", required=True)
    parser.add_argument("--input-prompts-dir", required=True)
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--output-prompts-dir", required=True)
    parser.add_argument("--per-category", type=int, default=25)
    parser.add_argument("--total", type=int, default=None, help="Approximate total item_id budget")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def first_existing_prompt(prompts_dir: Path, item_id: str) -> Path | None:
    candidates = [prompts_dir / f"{item_id}.txt"]
    try:
        candidates.append(prompts_dir / f"{int(float(item_id))}.txt")
    except ValueError:
        pass
    for path in candidates:
        if path.is_file():
            return path
    return None


def main() -> None:
    args = parse_args()
    csv_path = Path(args.input_csv)
    prompts_dir = Path(args.input_prompts_dir)
    out_csv = Path(args.output_csv)
    out_prompts = Path(args.output_prompts_dir)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out_prompts.mkdir(parents=True, exist_ok=True)

    with csv_path.open(newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        fieldnames = reader.fieldnames

    if not fieldnames:
        raise SystemExit(f"empty DPG csv: {csv_path}")
    if "item_id" not in fieldnames:
        raise SystemExit(f"DPG csv must contain item_id: {csv_path}")
    if "category_broad" not in fieldnames:
        raise SystemExit(f"DPG csv must contain category_broad: {csv_path}")

    by_item: dict[str, list[dict[str, str]]] = collections.defaultdict(list)
    item_categories: dict[str, set[str]] = collections.defaultdict(set)
    for row in rows:
        item_id = str(row["item_id"])
        by_item[item_id].append(row)
        category = row.get("category_broad", "")
        if category:
            item_categories[item_id].add(category)

    by_category: dict[str, list[str]] = collections.defaultdict(list)
    for item_id, categories in item_categories.items():
        for category in categories:
            by_category[category].append(item_id)

    rng = random.Random(args.seed)
    chosen_ids: list[str] = []
    chosen_set: set[str] = set()
    categories = sorted(by_category)
    per_category = args.per_category
    if args.total is not None:
        per_category = max(1, (args.total + len(categories) - 1) // len(categories))

    for category in categories:
        item_ids = list(by_category[category])
        if len(item_ids) < per_category and args.total is None:
            raise SystemExit(f"not enough DPG prompts for {category}: {len(item_ids)} < {per_category}")
        rng.shuffle(item_ids)
        added = 0
        for item_id in item_ids:
            if item_id in chosen_set:
                continue
            chosen_ids.append(item_id)
            chosen_set.add(item_id)
            added += 1
            if added >= per_category:
                break
            if args.total is not None and len(chosen_ids) >= args.total:
                break
        if args.total is not None and len(chosen_ids) >= args.total:
            break

    if args.total is not None and len(chosen_ids) < args.total:
        remaining = [item_id for item_id in by_item if item_id not in chosen_set]
        rng.shuffle(remaining)
        for item_id in remaining[: args.total - len(chosen_ids)]:
            chosen_ids.append(item_id)
            chosen_set.add(item_id)

    if args.total is not None:
        chosen_ids = chosen_ids[: args.total]
    rng.shuffle(chosen_ids)

    for old_file in out_prompts.glob("*.txt"):
        old_file.unlink()

    missing = []
    for item_id in chosen_ids:
        src = first_existing_prompt(prompts_dir, item_id)
        if src is None:
            missing.append(item_id)
            continue
        shutil.copy2(src, out_prompts / f"{item_id}.txt")
    if missing:
        raise SystemExit(f"missing DPG prompt txt files for item_id: {', '.join(missing[:10])}")

    chosen_set = set(chosen_ids)
    chosen_rows = [row for row in rows if str(row["item_id"]) in chosen_set]
    with out_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(chosen_rows)

    print(f"[dpg prompts] {len(chosen_ids)} item_ids -> {out_prompts}")
    print(f"[dpg prompts] {len(chosen_rows)} csv rows -> {out_csv}")
    covered = collections.Counter()
    for item_id in chosen_ids:
        covered.update(item_categories[item_id])
    for category in categories:
        print(f"  {category}: {covered[category]}")


if __name__ == "__main__":
    main()
