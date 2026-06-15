"""Local watchdog: poll SLURM job state every 5min, resubmit if dead.

Exits when remote log contains ``ALL DONE``. Replaces ``watchdog.sh``.

Usage:
    python scripts/watchdog.py <initial-jobid> [--host hse-hpc]
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from datetime import datetime


SBATCH_PATH = "~/geneval/scripts/run_strat.sbatch"
DONE_LOG_GLOB = "~/geneval/logs/strat_*.out"

ALIVE_STATES = {
    "RUNNING", "PENDING", "COMPLETING", "REQUEUED",
    "RESIZING", "SUSPENDED", "CONFIGURING",
}
BAD_STATES = {
    "CANCELLED", "CANCELLED+", "FAILED", "TIMEOUT", "NODE_FAIL",
    "OUT_OF_MEMORY", "BOOT_FAIL", "DEADLINE", "PREEMPTED",
}


def log(msg):
    print(f"[{datetime.now():%F %T}] {msg}", flush=True)


def ssh(host, cmd):
    return subprocess.run(
        ["ssh", host, cmd], capture_output=True, text=True
    )


def query_state(host, jobid):
    r = ssh(host, f"sacct -j {jobid} --format=State -P --noheader 2>/dev/null | head -1 | awk '{{print $1}}'")
    s = r.stdout.strip().split("+")[0] if r.returncode == 0 else ""
    return s or "QUERY_FAIL"


def all_done(host):
    r = ssh(host, f"grep -q 'ALL DONE' {DONE_LOG_GLOB} 2>/dev/null && echo YES || echo NO")
    return r.stdout.strip().startswith("YES")


def resubmit(host):
    r = ssh(host, f"sbatch --parsable {SBATCH_PATH}")
    new = r.stdout.strip()
    if not new:
        log("sbatch returned empty; retry next tick")
        return None
    log(f"resubmitted as {new}")
    return new


def main():
    p = argparse.ArgumentParser()
    p.add_argument("jobid")
    p.add_argument("--host", default="hse-hpc")
    p.add_argument("--interval", type=int, default=300)
    args = p.parse_args()

    jobid = args.jobid
    while True:
        state = query_state(args.host, jobid)
        log(f"jobid={jobid} state={state}")

        if state in ALIVE_STATES:
            pass
        elif state == "COMPLETED":
            if all_done(args.host):
                log("ALL DONE detected. exit.")
                return
            log("COMPLETED but no ALL DONE -> resubmit")
            new = resubmit(args.host)
            if new:
                jobid = new
        elif state.split("+")[0] in BAD_STATES or state in BAD_STATES:
            log(f"bad terminal state -> resubmit")
            new = resubmit(args.host)
            if new:
                jobid = new
        else:
            log(f"unknown state {state!r}; skipping for safety")

        time.sleep(args.interval)


if __name__ == "__main__":
    main()
