"""P2 cluster test: JobHandle + SlurmPool against real SLURM (no GPU compute needed).

  --jobid <id>   probe an existing job: state/node + a srun --overlap echo step
  --full         submit a short sbatch holder, wait running, srun a step, then scancel

Run on the login node (cwd=/path/to/inference):
    python -m cfgbench.tests.p2_slurm --jobid 4102912 --full --walltime 0:10:00
"""

from __future__ import annotations

import argparse
import tempfile
from pathlib import Path

from cfgbench.core.logsetup import configure
from inference.cfgbench.core.slurm import JobHandle, PoolConfig, SlurmPool, wall_seconds


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobid", help="existing job to probe")
    ap.add_argument("--full", action="store_true", help="submit+step+cancel a short holder")
    ap.add_argument("--walltime", default="0:10:00")
    a = ap.parse_args()
    log = configure(verbose=True)

    assert wall_seconds("0:10:00") == 600 and wall_seconds("1-00:00:00") == 86400
    log.info("wall_seconds OK")

    if a.jobid:
        h = JobHandle(a.jobid)
        log.info("probe %s: state=%s node=%s alive=%s", a.jobid, h.state(), h.node, h.is_alive())
        rc = h.run_step("echo STEP_OK host=$(hostname); nvidia-smi -L 2>/dev/null | head -1")
        log.info("run_step rc=%s", rc)

    if a.full:
        cfg = PoolConfig(walltime=a.walltime, max_jobs=1)
        pool = SlurmPool(cfg, Path(tempfile.mkdtemp(prefix="cfgbench_pool_")), log=log)
        try:
            h = pool.allocate(start_timeout=180)
            log.info("allocated jobid=%s node=%s running=%s", h.jobid, h.node, h.is_running())
            poolfile = pool.log_dir / "pool.json"
            pool.save(poolfile)
            log.info("pool.json: %s", poolfile.read_text().replace(chr(10), " "))
            if h.is_running():
                rc = h.run_step("echo HOLDER_STEP_OK host=$(hostname)")
                log.info("holder step rc=%s", rc)
            # reattach round-trip
            pool2 = SlurmPool(cfg, pool.log_dir, log=log)
            re = pool2.reattach(poolfile)
            log.info("reattach found %d live job(s)", len(re))
        finally:
            pool.shutdown()
            log.info("shutdown done")


if __name__ == "__main__":
    main()
