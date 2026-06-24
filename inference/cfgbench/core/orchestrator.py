"""Autonomous campaign orchestrator: manifest → shards → SlurmPool, with health + rotation.

Loop (resumable, idempotent — all state derived from the filesystem):
  1. expand campaign → manifest; reattach/keep the pool at max_jobs holder jobs.
  2. plan pending shards from the FS (gen per (model,config,bench,category); eval per
     (model,config,bench) once its gen is complete).
  3. dispatch shards onto free RUNNING jobs (one srun --overlap step per job, in a thread).
  4. health-check; on job death the pool reaps + replaces, and the shard's remaining items
     are simply replanned next round (the worker skips already-done items).
Continues until no gen/eval work remains.
"""

from __future__ import annotations

import shlex
import sys
import threading
import time
from collections import defaultdict

from ..config import CampaignSpec
from . import manifest as M
from .events import EventLog
from .health import HealthChecker
from .layout import CampaignPaths, eval_done, gen_done, gen_failed
from .logsetup import configure
from .shard import Shard
from .slurm import PoolConfig, SlurmPool, step_pre


def plan_shards(spec: CampaignSpec, paths: CampaignPaths, refs: list) -> list:
    """Pending gen shards (per category) + ready eval shards (per bench)."""
    want_gen = spec.stage in {"generate", "run"}
    want_eval = spec.stage in {"validate", "run"}
    gen_by = defaultdict(list)
    eval_by = defaultdict(list)
    for r in refs:
        if gen_done(paths, r):
            if want_eval:
                eval_by[(r.model, r.config, r.bench)].append(r)
        elif want_gen and not gen_failed(paths, r):
            gen_by[(r.model, r.config, r.bench, r.category)].append(r)

    shards = []
    for (m, c, b, cat), items in sorted(gen_by.items()):
        shards.append(Shard(m, c, b, "gen", [i.item_id for i in items], str(paths.root),
                            category=cat, seed=spec.seed, verbose=spec.verbose))
    for (m, c, b), items in sorted(eval_by.items()):
        if all(gen_done(paths, r) for r in items) and any(not eval_done(paths, r) for r in items):
            shards.append(Shard(m, c, b, "eval", [i.item_id for i in items], str(paths.root),
                                seed=spec.seed, verbose=spec.verbose))
    return shards


class Orchestrator:
    def __init__(self, spec: CampaignSpec, poll: float = 20):
        self.spec = spec
        self.poll = poll
        self.paths = CampaignPaths(spec.out_root).ensure()
        self.events = EventLog(self.paths)
        self.log = configure(spec.verbose, logfile=self.paths.orchestrator_log(),
                             name="cfgbench.orch")
        self.pool = SlurmPool(PoolConfig.from_dict(spec.pool), self.paths.logs,
                              events=self.events, log=self.log)
        self.health = HealthChecker(self.paths, self.pool.cfg.max_jobs,
                                    events=self.events, log=self.log)
        self._active: dict = {}        # jobid -> shard.name
        self._inflight: set = set()    # shard.name
        self._threads: list = []
        self._lock = threading.Lock()
        self._shards_dir = self.paths.root / "_shards"
        self._shards_dir.mkdir(parents=True, exist_ok=True)

    def _prompts_for(self, bench):
        from ..benchmarks.registry import get_benchmark
        return get_benchmark(bench).prompts()

    def run(self):
        items = M.expand(self.spec, self._prompts_for)
        M.write_manifest(self.paths, items)
        refs = [it.ref for it in items]
        self.events.emit("campaign_start", name=self.spec.name, items=len(refs),
                         models=",".join(self.spec.models), benchmarks=",".join(self.spec.benchmarks),
                         stage=self.spec.stage)
        self.log.info("campaign %s: %d items, stage=%s, max_jobs=%d",
                      self.spec.name, len(refs), self.spec.stage, self.pool.cfg.max_jobs)
        self.pool.reattach(self.paths.pool)
        try:
            self._loop(refs)
        finally:
            self._join_active(timeout=5)
            self.pool.save(self.paths.pool)
        st = M.scan_status(self.paths, refs)
        self.events.emit("campaign_done", gen_done=st.gen_done, eval_done=st.eval_done,
                         total=st.total, failed=st.failed)
        self.log.info("DONE: gen=%d eval=%d failed=%d / %d",
                      st.gen_done, st.eval_done, st.failed, st.total)
        self.pool.shutdown()          # release holder jobs on clean completion
        self.pool.save(self.paths.pool)
        return st

    def _loop(self, refs):
        while True:
            st = M.scan_status(self.paths, refs)
            shards = plan_shards(self.spec, self.paths, refs)
            if not shards:
                break

            self.pool.ensure()
            self.pool.save(self.paths.pool)

            with self._lock:
                inflight = set(self._inflight)
                busy = set(self._active)
            todo = [s for s in shards if s.name not in inflight]
            free = [j for j in self.pool.running() if j.jobid not in busy]
            for job in free:
                if not todo:
                    break
                self._dispatch(job, todo.pop(0))

            self.health.check(st=st, pool_live=len(self.pool.live()),
                              pool_running=len(self.pool.running()))
            self.log.info("gen=%d/%d eval=%d/%d failed=%d | jobs live=%d run=%d active=%d | pending_shards=%d",
                          st.gen_done, st.total, st.eval_done, st.total, st.failed,
                          len(self.pool.live()), len(self.pool.running()),
                          len(self._active), len(todo))
            self._reap_threads()
            time.sleep(self.poll)

    def _dispatch(self, job, shard):
        path = shard.save(self._shards_dir / f"{shard.name}.json")
        cmd = (
            f"{step_pre()}; "
            f"{shlex.quote(sys.executable)} -m cfgbench.cli worker --shard {shlex.quote(str(path))}"
        )
        logf = self.paths.worker_log(job.jobid, shard.name)
        with self._lock:
            self._active[job.jobid] = shard.name
            self._inflight.add(shard.name)
        self.events.emit("shard_start", jobid=job.jobid, node=job.node,
                         shard=shard.name, n=len(shard.item_ids))

        def target():
            rc = 1
            try:
                rc = job.run_step(cmd, log=logf)
            except Exception as e:
                self.log.error("step error %s: %s", shard.name, e)
            self.events.emit("shard_done" if rc == 0 else "shard_fail",
                             level="INFO" if rc == 0 else "WARNING",
                             jobid=job.jobid, shard=shard.name, rc=rc)
            with self._lock:
                self._active.pop(job.jobid, None)
                self._inflight.discard(shard.name)

        t = threading.Thread(target=target, daemon=True)
        t.start()
        with self._lock:
            self._threads.append(t)

    def _reap_threads(self):
        with self._lock:
            self._threads = [t for t in self._threads if t.is_alive()]

    def _join_active(self, timeout=None):
        for t in list(self._threads):
            t.join(timeout=timeout)
