"""Health checker: detect cluster/run anomalies → event log + ALERT sentinel + Telegram.

Runs inside the orchestrator loop (~every poll) and standalone as ``cfgbench health``
(non-zero exit on any anomaly, cron-friendly). Never trusts a single signal.
"""

from __future__ import annotations

import shutil
import time
from dataclasses import dataclass
from pathlib import Path

from . import notify
from .layout import CampaignPaths


@dataclass
class Alert:
    level: str   # WARNING | CRITICAL
    kind: str
    msg: str


@dataclass
class HealthThresholds:
    starve_s: float = 1200      # pool below target this long → starvation
    stall_s: float = 1800       # no progress with running jobs → stall
    disk_soft_gb: float = 10.0
    disk_hard_gb: float = 3.0
    fail_rate: float = 0.3
    min_fail: int = 5


class HealthChecker:
    def __init__(self, paths, target_jobs, events=None, log=None,
                 thresholds=None, notify_on_critical=True):
        self.paths = paths if isinstance(paths, CampaignPaths) else CampaignPaths(paths)
        self.target = target_jobs
        self.events = events
        self.log = log
        self.t = thresholds or HealthThresholds()
        self.notify_on_critical = notify_on_critical
        self._progress = (time.time(), None)
        self._starve_since = None
        self._fired: set = set()

    @staticmethod
    def _disk_free_gb():
        try:
            return shutil.disk_usage(Path.home()).free / 1e9
        except OSError:
            return None

    def check(self, *, st, pool_live, pool_running) -> list:
        now = time.time()
        pending = bool(st.pending_gen) or bool(st.pending_eval)
        alerts: list = []

        cur = (st.gen_done, st.eval_done)
        if self._progress[1] != cur:
            self._progress = (now, cur)
        idle = now - self._progress[0]

        if pending and pool_running and idle > self.t.stall_s:
            alerts.append(Alert("CRITICAL", "stall",
                                f"no progress for {int(idle)}s with {pool_running} running job(s)"))

        if pending and pool_live < self.target:
            self._starve_since = self._starve_since or now
            waited = now - self._starve_since
            if waited > self.t.starve_s:
                alerts.append(Alert("CRITICAL", "starvation",
                                    f"only {pool_live}/{self.target} jobs for {int(waited)}s"))
        else:
            self._starve_since = None

        free = self._disk_free_gb()
        if free is not None:
            if free < self.t.disk_hard_gb:
                alerts.append(Alert("CRITICAL", "disk_low", f"{free:.1f}GB free in $HOME"))
            elif free < self.t.disk_soft_gb:
                alerts.append(Alert("WARNING", "disk_low", f"{free:.1f}GB free in $HOME"))

        if st.total and st.failed >= self.t.min_fail and st.failed / st.total >= self.t.fail_rate:
            alerts.append(Alert("CRITICAL", "failure_spike",
                                f"{st.failed}/{st.total} items failed"))

        self._dispatch(alerts)
        return alerts

    def _dispatch(self, alerts: list) -> None:
        for a in alerts:
            if self.events:
                self.events.emit("alert", level=a.level, alert_kind=a.kind, msg=a.msg)
            if self.log:
                (self.log.error if a.level == "CRITICAL" else self.log.warning)(
                    "ALERT %s: %s", a.kind, a.msg)

        crit = [a for a in alerts if a.level == "CRITICAL"]
        if not crit:
            self._clear_sentinel()
            return

        self.paths.alert.write_text("\n".join(f"{a.kind}: {a.msg}" for a in crit) + "\n")
        new = [a for a in crit if a.kind not in self._fired]
        if new and self.notify_on_critical:
            text = ("cfgbench CRITICAL — campaign " + self.paths.root.name + ":\n"
                    + "\n".join(f"- {a.kind}: {a.msg}" for a in crit))
            ok = notify.send(text)
            for a in new:
                self._fired.add(a.kind)
            if self.events:
                self.events.emit("notify", ok=ok, kinds=[a.kind for a in new])

    def _clear_sentinel(self) -> None:
        try:
            self.paths.alert.unlink()
        except FileNotFoundError:
            pass
        self._fired.clear()


def check_once(campaign) -> int:
    """Standalone check: load a campaign, scan once, report. Returns exit code (0 ok, 1 anomaly)."""
    from ..benchmarks.registry import get_benchmark
    from ..config import load_campaign
    from . import manifest as M
    from .slurm import PoolConfig, SlurmPool

    spec = load_campaign(campaign)
    paths = CampaignPaths(spec.out_root)
    refs = M.load_manifest(paths) or [it.ref for it in M.expand(spec, lambda b: get_benchmark(b).prompts())]
    st = M.scan_status(paths, refs)

    pool = SlurmPool(PoolConfig.from_dict(spec.pool), paths.logs)
    pool.reattach(paths.pool)
    live, running = len(pool.live()), len(pool.running())

    checker = HealthChecker(paths, pool.cfg.max_jobs, notify_on_critical=False)
    alerts = checker.check(st=st, pool_live=live, pool_running=running)

    print(f"campaign={spec.name} gen={st.gen_done}/{st.total} eval={st.eval_done}/{st.total} "
          f"failed={st.failed} jobs(live={live},run={running}) alert={'YES' if paths.alert.is_file() else 'no'}")
    for a in alerts:
        print(f"  [{a.level}] {a.kind}: {a.msg}")
    return 1 if alerts else 0
