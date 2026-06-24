"""SLURM allocation + step execution for the worker pool.

Mechanism (Slurm 24.11.7 on this cluster has no ``salloc --no-shell``):
  * hold an allocation with an **sbatch holder** job running ``sleep <walltime>`` — fully
    detached, survives orchestrator/SSH death (autonomy);
  * run work via ``srun --jobid=<id> --overlap`` steps into the holder.
``gpus_per_job=1`` everywhere → the lone GPU is index 0, sidestepping the
CUDA_VISIBLE_DEVICES/GRES pinning bug on multi-GPU steps.
"""

from __future__ import annotations

import json
import re
import shlex
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

def step_pre(repo_root=None, venv=None) -> str:
    """Shell preamble for every srun step: new glibc, venv activate, real TMPDIR, repo cwd.

    Defaults derive from the running orchestrator (its own venv via ``sys.prefix`` and
    repo location), so this works regardless of ``$HOME`` / venv / clone path — important
    for sharing with another account on the same cluster.
    """
    import sys
    if venv is None:
        venv = sys.prefix
    if repo_root is None:
        repo_root = Path(__file__).resolve().parent.parent.parent
    venv = Path(venv)
    repo_root = Path(repo_root)
    activate = shlex.quote(str(venv / "bin" / "activate"))
    venv_bin = shlex.quote(str(venv / "bin"))
    cwd = shlex.quote(str(repo_root))
    return (
        "module load gnu14/14.1 2>/dev/null; "
        f"export VIRTUAL_ENV={shlex.quote(str(venv))}; "
        f"export PATH={venv_bin}:$PATH; "
        f"source {activate} 2>/dev/null || true; "
        "if [ -z \"${TMPDIR:-}\" ] || [ \"${TMPDIR:-}\" = \"$HOME/.tmp\" ]; then "
        "if [ -n \"${SLURM_TMPDIR:-}\" ]; then export TMPDIR=\"$SLURM_TMPDIR\"; "
        "else export TMPDIR=\"/tmp/${USER:-cfgbench}/cfgbench-${SLURM_JOB_ID:-local}\"; fi; fi; "
        "mkdir -p \"$TMPDIR\"; "
        "if [ -n \"${CFG_HF_HOME:-}\" ] && [ -z \"${HF_HOME:-}\" ]; then export HF_HOME=\"$CFG_HF_HOME\"; fi; "
        "if [ -n \"${HF_HOME:-}\" ]; then mkdir -p \"$HF_HOME\"; fi; "
        f"cd {cwd}"
    )

_ALIVE = {"RUNNING", "PENDING", "CONFIGURING", "COMPLETING", "RESV_DEL_HOLD", "REQUEUED"}


def _sh(cmd, timeout=60):
    r = subprocess.run([str(c) for c in cmd], capture_output=True, text=True, timeout=timeout)
    return r.returncode, r.stdout.strip(), r.stderr.strip()


def wall_seconds(walltime: str) -> int:
    """'HH:MM:SS' or 'D-HH:MM:SS' → seconds."""
    days, t = 0, walltime
    if "-" in t:
        d, t = t.split("-", 1)
        days = int(d)
    parts = [int(x) for x in t.split(":")]
    while len(parts) < 3:
        parts.insert(0, 0)
    h, m, s = parts
    return days * 86400 + h * 3600 + m * 60 + s


@dataclass
class JobHandle:
    jobid: str
    node: str | None = None
    _state: str | None = None

    def refresh(self) -> str | None:
        rc, out, _ = _sh(["squeue", "-h", "-j", self.jobid, "-o", "%T|%N"])
        if rc != 0 or not out:
            self._state = None
            return None
        st, _, node = out.partition("|")
        self._state = st.strip()
        if node.strip():
            self.node = node.strip()
        return self._state

    def state(self) -> str | None:
        return self.refresh()

    def is_running(self) -> bool:
        return self.refresh() == "RUNNING"

    def is_alive(self) -> bool:
        return self.refresh() in _ALIVE

    def wait_running(self, timeout: float = 300, poll: float = 5) -> bool:
        t0 = time.time()
        while time.time() - t0 < timeout:
            st = self.refresh()
            if st == "RUNNING" and self.node:
                return True
            if st is None:
                return False  # vanished (rejected / cancelled)
            time.sleep(poll)
        return self.is_running()

    def run_step(self, command: str, log=None, timeout=None, extra_srun=None) -> int:
        srun = ["srun", "--jobid", self.jobid, "--overlap", "-N1", "-n1"]
        if extra_srun:
            srun += list(extra_srun)
        srun += ["bash", "-lc", command]
        srun = [str(c) for c in srun]
        if log:
            with open(log, "ab") as f:
                return subprocess.run(srun, stdout=f, stderr=subprocess.STDOUT, timeout=timeout).returncode
        return subprocess.run(srun, timeout=timeout).returncode

    def cancel(self) -> None:
        _sh(["scancel", self.jobid])


@dataclass
class PoolConfig:
    partition: str = "rocky"
    gpu_type: str = "v100"      # "" → any gpu
    gpus_per_job: int = 1
    cpus_per_gpu: int = 6
    walltime: str = "12:00:00"
    max_jobs: int = 4
    name: str = "cfgbench-w"
    account: str = ""   # SLURM -A (set if your profile requires an account)

    @staticmethod
    def from_dict(d: dict) -> "PoolConfig":
        d = d or {}
        return PoolConfig(
            partition=d.get("partition", "rocky"),
            gpu_type=d.get("gpu_type", "v100"),
            gpus_per_job=int(d.get("gpus_per_job", 1)),
            cpus_per_gpu=int(d.get("cpus_per_gpu", 6)),
            walltime=str(d.get("walltime", "12:00:00")),
            max_jobs=int(d.get("max_jobs", 4)),
            name=d.get("name", "cfgbench-w"),
            account=d.get("account", ""),
        )


class SlurmPool:
    """Maintains a set of live 1-GPU holder jobs; replaces dead ones; reattachable."""

    def __init__(self, cfg: PoolConfig, log_dir, events=None, log=None):
        self.cfg = cfg
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.events = events
        self.log = log
        self.jobs: dict[str, JobHandle] = {}

    def _emit(self, kind, level="INFO", **kw):
        if self.events:
            self.events.emit(kind, level=level, **kw)
        if self.log:
            self.log.info("%s %s", kind, kw)

    def _gres(self) -> str:
        g = self.cfg.gpus_per_job
        return f"gpu:{self.cfg.gpu_type}:{g}" if self.cfg.gpu_type else f"gpu:{g}"

    def submit_holder(self) -> JobHandle:
        cpus = self.cfg.cpus_per_gpu * self.cfg.gpus_per_job
        log = self.log_dir / "holder_%j.out"  # %j expanded by slurm
        cmd = ["sbatch", "-p", self.cfg.partition, "--gres", self._gres(), "-c", str(cpus),
               "-t", self.cfg.walltime, "-J", self.cfg.name, "-o", str(log),
               *(["-A", self.cfg.account] if self.cfg.account else []),
               "--wrap", f"sleep {wall_seconds(self.cfg.walltime)}"]
        rc, out, err = _sh(cmd)
        if rc != 0:
            raise RuntimeError(f"sbatch failed: {err or out}")
        m = re.search(r"Submitted batch job (\d+)", out)
        if not m:
            raise RuntimeError(f"cannot parse sbatch output: {out!r}")
        h = JobHandle(m.group(1))
        self.jobs[h.jobid] = h
        self._emit("job_submit", jobid=h.jobid, partition=self.cfg.partition, gres=self._gres())
        return h

    def allocate(self, start_timeout: float = 300) -> JobHandle:
        """Submit a holder and wait for it to start (returns even if still pending)."""
        h = self.submit_holder()
        if h.wait_running(start_timeout):
            self._emit("job_alloc", jobid=h.jobid, node=h.node)
        else:
            self._emit("job_pending", level="WARNING", jobid=h.jobid)
        return h

    def prune_dead(self) -> list[str]:
        dead = [jid for jid, h in self.jobs.items() if not h.is_alive()]
        for jid in dead:
            self._emit("job_dead", level="WARNING", jobid=jid)
            self.jobs.pop(jid, None)
        return dead

    def ensure(self, n: int | None = None) -> list[JobHandle]:
        """Top the pool up to ``n`` jobs (default max_jobs); reap dead first.

        Non-blocking: submits holders (PENDING) and returns immediately — newly
        submitted jobs become usable once ``running()`` reports them RUNNING.
        """
        n = n or self.cfg.max_jobs
        self.prune_dead()
        while len(self.jobs) < n:
            self.submit_holder()
        return list(self.jobs.values())

    def live(self) -> list[JobHandle]:
        return [h for h in self.jobs.values() if h.is_alive()]

    def running(self) -> list[JobHandle]:
        return [h for h in self.jobs.values() if h.is_running()]

    def save(self, path) -> None:
        Path(path).write_text(json.dumps(
            {"name": self.cfg.name, "jobids": sorted(self.jobs)}, indent=2))

    def reattach(self, path) -> list[JobHandle]:
        p = Path(path)
        if not p.is_file():
            return []
        for jid in json.loads(p.read_text()).get("jobids", []):
            h = JobHandle(str(jid))
            if h.is_alive():
                self.jobs[h.jobid] = h
                self._emit("job_reattach", jobid=h.jobid, node=h.node)
        return list(self.jobs.values())

    def shutdown(self) -> None:
        for h in list(self.jobs.values()):
            h.cancel()
            self._emit("job_cancel", jobid=h.jobid)
        self.jobs.clear()
