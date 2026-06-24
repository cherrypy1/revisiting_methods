"""Unattended overnight driver: wait for cosmos2 generations to finish, fill any
gen gaps on the two long-lived jobs, then run DPG (mplug) + OneIG evaluation for
all methods in parallel across those two GPUs, and finally write result tables.

Runs on the LOGIN node (nohup); it only does filesystem polling + ``srun
--jobid --overlap`` dispatch, so it needs no GPU itself. Designed to be launched
and left alone — the user is asleep.

Layout assumptions (TAG = cosmos2_24052026):
    outputs/cosmos2/dpg/<m>_<TAG>/images/*.png         (90 per method)
    outputs/cosmos2/oneig/<TAG>/{object,text,reasoning}/<m>/*.webp  (15 each)

Jobs at launch time:
    4070630 (cn-010, 2 GPU, ~5h left) — still generating: oseg DPG + no_cfg OneIG
    4072690 (cn-001, 1 GPU, ~11h)     — finishing oseg OneIG, then free for eval
    4072693 (cn-006, 1 GPU, ~11h)     — finishing pag  OneIG, then free for eval
"""

from __future__ import annotations

import subprocess
import threading
import time
from datetime import datetime
from pathlib import Path

HOME = Path.home()
REPO = HOME / "geneval"
OUT = REPO / "outputs" / "cosmos2"
LOGS = REPO / "logs"
TAG = "cosmos2_24052026"
METHODS = ["no_cfg", "cfg", "cfgpp", "cfg0s", "apg",
           "tcfg", "sag", "seg_sigma10", "oseg", "pag"]
CATS = ["object", "text", "reasoning"]

SHORT_JOB = "4070630"                 # the expiring 2-GPU job
EVAL_JOBS = ["4072690", "4072693"]    # the two long jobs we run eval on
FREE_WHEN = {"4072690": "oseg", "4072693": "pag"}  # job free after this OneIG gen

PRE = ("module load gnu14/14.1 >/dev/null 2>&1; source ~/.venv/bin/activate; "
       "export TMPDIR=$HOME/.tmp; mkdir -p $HOME/.tmp; cd ~/geneval")

MASTER = LOGS / "overnight.out"


def log(msg):
    line = f"[{datetime.now():%Y-%m-%d %H:%M:%S}] {msg}"
    print(line, flush=True)
    with MASTER.open("a") as f:
        f.write(line + "\n")


def count_dpg(m):
    return len(list((OUT / "dpg" / f"{m}_{TAG}" / "images").glob("*.png")))


def count_oneig(m):
    return sum(len(list((OUT / "oneig" / TAG / c / m).glob("*.webp"))) for c in CATS)


def alive(job):
    r = subprocess.run(["squeue", "-h", "-j", job],
                       capture_output=True, text=True)
    return bool(r.stdout.strip())


def srun(job, shell_cmd, logfile, port=None):
    """Blocking srun step inside a job; full output to logfile."""
    pre = PRE + (f"; export MASTER_PORT={port}" if port else "")
    full = ["srun", "--jobid", job, "--overlap", "bash", "-lc",
            f"{pre}; {shell_cmd}"]
    with open(logfile, "w") as fh:
        return subprocess.run(full, stdout=fh, stderr=subprocess.STDOUT).returncode


# ---- command builders -------------------------------------------------------

def gen_dpg_cmd(m):
    return (f"python scripts/bench.py {m} dpg --model cosmos2 --run-tag {TAG} "
            f"--limit 90 --pic-num 1 --skip-eval")


def gen_oneig_cmd(m):
    return (f"python scripts/bench.py {m} oneig --model cosmos2 --run-tag {TAG} "
            f"--limit 15 --grid 1x1 --skip-eval")


def eval_dpg_cmd(m, port):
    return (f"python scripts/eval_dpg.py --dpg-root $HOME/ELLA "
            f"--image-dir outputs/cosmos2/dpg/{m}_{TAG}/images "
            f"--pic-num 1 --resolution 1024 "
            f"--out-dir outputs/cosmos2/dpg/{m}_{TAG}/eval --port {port}")


def eval_oneig_cmd(m):
    return (f"python scripts/eval_oneig.py --oneig-root $HOME/OneIG-Benchmark "
            f"--image-dir outputs/cosmos2/oneig/{TAG} --model-name {m} "
            f"--grid 1x1 --out-dir outputs/cosmos2/oneig/{TAG}/eval/{m}")


# ---- waiting ---------------------------------------------------------------

def wait_until(pred, label, deadline_h):
    deadline = time.time() + deadline_h * 3600
    while not pred():
        if time.time() > deadline:
            log(f"WAIT timeout ({deadline_h}h) for: {label} — proceeding anyway")
            return
        time.sleep(300)
    log(f"ready: {label}")


# ---- parallel run across the two eval jobs ---------------------------------

def run_tasks_parallel(tasks):
    """tasks = list of (label, cmd_for_job_fn). Round-robin to the 2 jobs;
    each job runs its slice sequentially. cmd_for_job_fn(port)->shell_cmd."""
    buckets = {j: [] for j in EVAL_JOBS}
    for i, t in enumerate(tasks):
        buckets[EVAL_JOBS[i % len(EVAL_JOBS)]].append(t)

    def worker(job, slice_, port):
        for label, cmd_fn in slice_:
            lf = LOGS / f"overnight_{label}_{job}.out"
            log(f"[{job}] start {label}")
            rc = srun(job, cmd_fn(port), lf, port=port)
            log(f"[{job}] done  {label} rc={rc} -> {lf.name}")

    threads = []
    for k, job in enumerate(EVAL_JOBS):
        t = threading.Thread(target=worker,
                             args=(job, buckets[job], 29500 + 10 * k))
        t.start()
        threads.append(t)
    for t in threads:
        t.join()


def main():
    LOGS.mkdir(exist_ok=True)
    log("=== overnight eval driver start ===")

    # 1) wait for the two eval jobs to finish their own OneIG generation
    wait_until(lambda: all(count_oneig(FREE_WHEN[j]) >= 45 for j in EVAL_JOBS),
               "eval jobs free (oseg+pag OneIG gen done)", deadline_h=6)

    # 2) wait for the expiring short job to finish OR die (its outputs settle,
    #    avoids racing on {idx}.png if we have to resume its gen)
    wait_until(lambda: (not alive(SHORT_JOB))
               or (count_dpg("oseg") >= 90 and count_oneig("no_cfg") >= 45),
               "short job settled (oseg DPG + no_cfg OneIG done, or job gone)",
               deadline_h=6)

    # 3) gap-fill any incomplete generation on the two long GPUs (resumable)
    gap_dpg = [m for m in METHODS if count_dpg(m) < 90]
    gap_oneig = [m for m in METHODS if count_oneig(m) < 45]
    log(f"gen gaps -> dpg:{gap_dpg} oneig:{gap_oneig}")
    gen_tasks = ([(f"gen_dpg_{m}", (lambda p, m=m: gen_dpg_cmd(m))) for m in gap_dpg]
                 + [(f"gen_oneig_{m}", (lambda p, m=m: gen_oneig_cmd(m))) for m in gap_oneig])
    if gen_tasks:
        run_tasks_parallel(gen_tasks)

    # 4) evaluate every method whose generation is complete
    eval_tasks = []
    for m in METHODS:
        if count_dpg(m) >= 90:
            eval_tasks.append((f"eval_dpg_{m}", (lambda p, m=m: eval_dpg_cmd(m, p))))
        else:
            log(f"SKIP dpg eval {m}: only {count_dpg(m)}/90 imgs")
        if count_oneig(m) >= 45:
            eval_tasks.append((f"eval_oneig_{m}", (lambda p, m=m: eval_oneig_cmd(m))))
        else:
            log(f"SKIP oneig eval {m}: only {count_oneig(m)}/45 imgs")
    log(f"eval tasks: {[t[0] for t in eval_tasks]}")
    run_tasks_parallel(eval_tasks)

    # 5) write result tables
    log("=== writing result tables ===")
    res = LOGS / "overnight_RESULTS.txt"
    with res.open("w") as fh:
        for title, cmd in [
            ("GENEVAL", ["python", "scripts/summarize_geneval.py", TAG,
                         "--model", "cosmos2", "--extra"]),
            ("ONEIG", ["python", "scripts/summarize_oneig.py", TAG,
                       "--model", "cosmos2", "--raw"]),
        ]:
            fh.write(f"\n########## {title} ##########\n")
            out = subprocess.run(cmd, cwd=REPO, capture_output=True, text=True)
            fh.write(out.stdout + out.stderr)
        # DPG has no summarizer yet — grep score lines from each eval log
        fh.write("\n########## DPG (raw score lines from eval logs) ##########\n")
        for m in METHODS:
            lf = LOGS / f"overnight_eval_dpg_{m}_*.out"
            for p in sorted(LOGS.glob(f"overnight_eval_dpg_{m}_*.out")):
                hits = [ln for ln in p.read_text(errors="ignore").splitlines()
                        if "score" in ln.lower() or "average" in ln.lower()]
                fh.write(f"{m}: {' | '.join(hits[-3:]) if hits else '(no score line)'}\n")
    log(f"results -> {res}")
    log("=== overnight eval driver DONE ===")


if __name__ == "__main__":
    main()
