"""Create a stratified GenEval JSONL subset."""

from __future__ import annotations

import argparse
import collections
import json
import random
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Full evaluation_metadata.jsonl")
    parser.add_argument("--output", required=True)
    parser.add_argument("--per-tag", type=int, default=25)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    src = Path(args.input)
    dst = Path(args.output)
    dst.parent.mkdir(parents=True, exist_ok=True)

    by_tag: dict[str, list[dict]] = collections.defaultdict(list)
    with src.open() as f:
        for line in f:
            item = json.loads(line)
            by_tag[item["tag"]].append(item)

    rng = random.Random(args.seed)
    chosen = []
    for tag in sorted(by_tag):
        items = list(by_tag[tag])
        if len(items) < args.per_tag:
            raise SystemExit(f"not enough GenEval prompts for {tag}: {len(items)} < {args.per_tag}")
        rng.shuffle(items)
        chosen.extend(items[: args.per_tag])

    rng.shuffle(chosen)
    with dst.open("w") as f:
        for item in chosen:
            f.write(json.dumps(item) + "\n")

    print(f"[geneval prompts] {len(chosen)} prompts -> {dst}")
    for tag in sorted(by_tag):
        print(f"  {tag}: {args.per_tag}")


if __name__ == "__main__":
    main()
