#!/usr/bin/env bash
# Usage: scripts/summarize_geneval.sh <run_tag>
# Walks outputs/geneval/*_<tag>/results.jsonl and prints upstream summary.
set -euo pipefail
TAG="${1:?run-tag required (e.g. final22042026)}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SUM="$ROOT/evaluation/summary_scores.py"

shopt -s nullglob
for d in "$ROOT"/outputs/geneval/*_"$TAG"; do
  f="$d/results.jsonl"
  [[ -f "$f" ]] || continue
  echo "########## $(basename "$d") ##########"
  python "$SUM" "$f"
  echo
done
