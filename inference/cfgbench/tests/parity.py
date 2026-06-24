"""P1 server parity harness: run real gen+eval for one (model, config, bench) on a small
stratified subset, through the generic worker, and print the resulting _summary.json.

Validates the adapters + worker end-to-end against the actual model and scorers (needs GPU).

Run (inside a GPU job, cwd=/path/to/inference):
    python -m cfgbench.tests.parity --bench geneval --limit 12
    python -m cfgbench.tests.parity --bench oneig  --limit 6
    python -m cfgbench.tests.parity --bench dpg    --limit 4
"""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

from cfgbench.benchmarks.registry import get_benchmark
from cfgbench.config import CampaignSpec
from cfgbench.core import manifest as M
from cfgbench.core.layout import CampaignPaths, read_json
from cfgbench.core.shard import Shard
from cfgbench.core.worker import run_shard


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="sd35")
    ap.add_argument("--config", default="cfg")
    ap.add_argument("--bench", required=True, choices=["geneval", "oneig", "dpg"])
    ap.add_argument("--limit", type=int, default=12)
    ap.add_argument("--samples", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    out = Path(a.out) if a.out else Path(tempfile.mkdtemp(prefix=f"cfgbench_parity_{a.bench}_"))
    spec = CampaignSpec(name="parity", models=[a.model], configs=[a.config], benchmarks=[a.bench],
                        out_root=out, samples_per_prompt={a.bench: a.samples},
                        limit={a.bench: a.limit}, sample_seed=0, seed=a.seed, verbose=True)
    items = M.expand(spec, lambda b: get_benchmark(b).prompts())
    paths = CampaignPaths(out).ensure()
    M.write_manifest(paths, items)
    ids = [it.ref.item_id for it in items]
    print(f"[parity] {a.model}/{a.config}/{a.bench}: {len(ids)} items -> {out}", flush=True)

    run_shard(Shard(a.model, a.config, a.bench, "gen", ids, str(out),
                    seed=a.seed, verbose=True).save(out / "gen.json"))
    run_shard(Shard(a.model, a.config, a.bench, "eval", ids, str(out),
                    verbose=True).save(out / "eval.json"))

    summ = read_json(paths.summary_bench(a.model, a.config, a.bench))
    print("[parity] SUMMARY:\n" + json.dumps(summ, indent=2, ensure_ascii=False))
    # show one sidecar so per-item metrics are visible
    side = read_json(paths.sidecar(items[0].ref))
    print("[parity] sample sidecar:\n" + json.dumps(side, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
