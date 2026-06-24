"""Generic shard executor — drives adapters; oblivious to model/bench internals.

Two phases:
  gen  — load the model adapter once, generate each pending sample, write image + sidecar JSON.
  eval — build GenItems for generated samples, call the bench adapter's evaluate + summarize,
         merge per-item metrics into sidecars and write _summary.json.

Idempotent (skips done items), per-item failure isolation, heartbeat + throughput logging.
"""

from __future__ import annotations

import os
import shutil
import time
from pathlib import Path

from ..benchmarks.base import GenItem
from ..benchmarks.registry import get_benchmark
from ..config import method_config_path
from ..models.registry import get_model
from .layout import (CampaignPaths, ItemRef, build_sidecar, eval_done, gen_done,
                     read_json, write_json_atomic)
from .logsetup import ThroughputMeter, configure
from .shard import Shard


def _jobid() -> str:
    return os.environ.get("SLURM_JOB_ID", "local")


def _heartbeat(paths: CampaignPaths, jobid: str, **kw) -> None:
    write_json_atomic(paths.heartbeat(jobid), {"ts": time.time(), **kw})


def run_shard(shard_path) -> None:
    shard = Shard.load(shard_path)
    paths = CampaignPaths(shard.campaign_out).ensure()
    jobid = _jobid()
    log = configure(shard.verbose, logfile=paths.worker_log(jobid, shard.name))
    log.info("shard %s phase=%s items=%d job=%s",
             shard.name, shard.phase, len(shard.item_ids), jobid)
    if shard.phase == "gen":
        _run_gen(shard, paths, log, jobid)
    elif shard.phase == "eval":
        _run_eval(shard, paths, log, jobid)
    else:
        raise ValueError(f"unknown phase: {shard.phase}")


def _run_gen(shard: Shard, paths: CampaignPaths, log, jobid: str) -> None:
    refs = [ItemRef.parse(i) for i in shard.item_ids]
    pending = [r for r in refs if not gen_done(paths, r)]
    log.info("gen: %d/%d pending", len(pending), len(refs))
    if not pending:
        return

    bench = get_benchmark(shard.bench)
    # key by (category, id): OneIG ids are unique only within a category
    pmap = {(p.category, p.id): p for p in bench.prompts()}
    cfg = method_config_path(shard.model, shard.config)
    if cfg is None:
        raise FileNotFoundError(f"no method config for {shard.model}/{shard.config}")
    try:
        handle = get_model(shard.model).load(cfg)
    except Exception as e:
        log.error("model load FAILED %s/%s: %s", shard.model, shard.config, e)
        for ref in pending:
            p = pmap.get((ref.category, ref.prompt_id))
            text = p.text if p is not None else ""
            side = build_sidecar(ref, text=text, seed=shard.seed + ref.sample_idx,
                                 params={}, gen_time_s=0.0, status="failed")
            side["error"] = f"model load failed: {e}"
            write_json_atomic(paths.sidecar(ref), side)
        return
    params = handle.params() if hasattr(handle, "params") else {}
    meter = ThroughputMeter()
    try:
        for n, ref in enumerate(pending, 1):
            p = pmap.get((ref.category, ref.prompt_id))
            if p is None:
                log.warning("no prompt for %s — skip", ref.prompt_id)
                continue
            seed = shard.seed + ref.sample_idx
            log.debug("gen %s (%d/%d)", ref.item_id, n, len(pending))
            t0 = time.time()
            try:
                img = handle.generate(p.text, seed=seed, n=1)[0]
            except Exception as e:  # isolate per-item failures
                log.error("gen FAILED %s: %s", ref.item_id, e)
                side = build_sidecar(ref, text=p.text, seed=seed, params=params,
                                     gen_time_s=0.0, status="failed")
                side["error"] = str(e)
                write_json_atomic(paths.sidecar(ref), side)
                continue
            dt = time.time() - t0

            ip = paths.image(ref)
            ip.parent.mkdir(parents=True, exist_ok=True)
            tmp = ip.with_name(ip.name + ".tmp")
            img.save(tmp, format="PNG")
            os.replace(tmp, ip)
            write_json_atomic(paths.sidecar(ref),
                              build_sidecar(ref, text=p.text, seed=seed, params=params,
                                            gen_time_s=round(dt, 3)))
            meter.tick()
            if n % 5 == 0 or shard.verbose:
                eta = meter.eta_s(len(pending) - n)
                log.info("gen %d/%d  %.3f img/s  eta=%ss",
                         n, len(pending), meter.rate(), int(eta) if eta else "?")
            _heartbeat(paths, jobid, shard=shard.name, phase="gen", item=ref.item_id,
                       done=n, total=len(pending), rate=round(meter.rate(), 3))
    finally:
        handle.close()
    log.info("gen done: %d images", meter.total)


def _run_eval(shard: Shard, paths: CampaignPaths, log, jobid: str) -> None:
    refs = [ItemRef.parse(i) for i in shard.item_ids]
    have = [r for r in refs if gen_done(paths, r)]
    if not have:
        log.warning("eval: no generated items")
        return

    bench = get_benchmark(shard.bench)
    pmap = {(p.category, p.id): p for p in bench.prompts()}
    items = [GenItem(r.item_id, pmap[(r.category, r.prompt_id)],
                     paths.image(r), paths.sidecar(r), r.sample_idx)
             for r in have if (r.category, r.prompt_id) in pmap]
    log.info("eval: %d items via %s", len(items), shard.bench)

    workdir = paths.root / "_evalwork" / shard.name
    shutil.rmtree(workdir, ignore_errors=True)
    workdir.mkdir(parents=True, exist_ok=True)
    _heartbeat(paths, jobid, shard=shard.name, phase="eval", total=len(items))

    metrics = bench.evaluate(items, workdir)

    for it in items:
        side = read_json(it.sidecar_path) or {}
        side["metrics"] = metrics.get(it.item_id, {})
        write_json_atomic(it.sidecar_path, side)

    summary = bench.summarize(metrics, [it.prompt for it in items])
    write_json_atomic(paths.summary_bench(shard.model, shard.config, shard.bench), summary)
    for cat, cell in (summary.get("by_category") or {}).items():
        write_json_atomic(paths.summary_category(shard.model, shard.config, shard.bench, cat), cell)
    log.info("eval done: overall=%s", summary.get("overall"))
    _heartbeat(paths, jobid, shard=shard.name, phase="eval", done=len(items), total=len(items))
