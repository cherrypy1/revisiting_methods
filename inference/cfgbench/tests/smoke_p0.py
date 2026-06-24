"""P0 local smoke: exercise the core with no SLURM / GPU / network.

Run: python -m cfgbench.tests.smoke_p0
"""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

from cfgbench.benchmarks.base import PromptSpec, default_summarize
from cfgbench.config import CampaignSpec, method_config_path
from cfgbench.core import manifest as M
from cfgbench.core import notify
from cfgbench.core.events import EventLog
from cfgbench.core.layout import (CampaignPaths, build_sidecar, eval_done,
                                  gen_done, read_json, write_json_atomic)
from cfgbench.core.logsetup import ThroughputMeter, configure


def main() -> None:
    log = configure(verbose=True)
    tmp = Path(tempfile.mkdtemp(prefix="cfgbench_smoke_"))
    try:
        paths = CampaignPaths(tmp).ensure()

        # --- atomic json ---
        write_json_atomic(tmp / "x.json", {"a": 1})
        assert read_json(tmp / "x.json") == {"a": 1}
        assert read_json(tmp / "nope.json") is None
        log.info("layout: atomic json OK")

        # --- events ---
        ev = EventLog(paths)
        ev.emit("campaign_start", name="smoke")
        ev.emit("job_alloc", jobid=123, node="cn-001")
        ev.emit("job_dead", level="WARNING", jobid=123)
        assert len(ev.read()) == 3
        assert paths.events_log.is_file()
        log.info("events: 3 lifecycle events OK")

        # --- throughput meter ---
        tm = ThroughputMeter(window_s=5)
        tm.tick(10)
        assert tm.total == 10
        assert tm.rate() >= 0.0
        log.info("logsetup: throughput meter OK")

        # --- notify (no creds -> graceful False, never raises) ---
        assert notify.available() in (True, False)
        assert notify.send("smoke ping") in (True, False)
        log.info("notify: graceful no-cred path OK (available=%s)", notify.available())

        # --- method config resolution against real repo configs ---
        assert method_config_path("sd35", "cfg") is not None, "configs/sd35/cfg.yaml expected"
        assert method_config_path("sd35", "does_not_exist") is None
        log.info("config: method_config_path OK")

        # --- manifest expand / write / load ---
        spec = CampaignSpec(name="smoke", models=["sd35"], configs=["cfg", "ghost"],
                            benchmarks=["dummy"], out_root=tmp,
                            samples_per_prompt={"dummy": 2}, seed=0)
        prompts = [PromptSpec(id=f"{i:05d}", text=f"prompt {i}",
                              category=("catA" if i % 2 else "catB")) for i in range(3)]
        items = M.expand(spec, lambda b: prompts)
        # "ghost" config has no yaml -> skipped; 3 prompts x 2 samples x 1 config = 6
        assert len(items) == 6, len(items)
        M.write_manifest(paths, items)
        refs = M.load_manifest(paths)
        assert len(refs) == 6
        assert refs[0].item_id == items[0].ref.item_id
        log.info("manifest: expand(6)/write/load round-trip OK")

        # --- status derived from filesystem ---
        st = M.scan_status(paths, refs)
        assert st.total == 6 and st.gen_done == 0 and len(st.pending_gen) == 6

        # simulate generating one item: image + sidecar -> gen_done True
        ref = refs[0]
        img = paths.image(ref)
        img.parent.mkdir(parents=True, exist_ok=True)
        try:
            from PIL import Image
            Image.new("RGB", (4, 4)).save(img)
        except ImportError:
            img.write_bytes(b"x")  # image_ok falls back to size>0 without PIL
        side = build_sidecar(ref, text="prompt 0", seed=0, params={"guidance_scale": 5.0},
                             gen_time_s=1.23)
        write_json_atomic(paths.sidecar(ref), side)
        assert gen_done(paths, ref) and not eval_done(paths, ref)

        # fill metrics -> eval_done True
        side["metrics"] = {"correct": 1}
        write_json_atomic(paths.sidecar(ref), side)
        assert eval_done(paths, ref)

        st2 = M.scan_status(paths, refs)
        assert st2.gen_done == 1 and st2.eval_done == 1 and len(st2.pending_gen) == 5
        log.info("status: idempotent FS-derived state OK (gen=1 eval=1 pending=5)")

        # --- default summarize ---
        metrics = {refs[i].item_id: {"correct": (1 if i < 3 else 0)} for i in range(6)}
        summ = default_summarize(metrics, prompts)
        assert "by_category" in summ and "overall" in summ and summ["n"] == 6
        log.info("summary: default_summarize OK %s", summ["overall"])

        log.info("P0 SMOKE PASSED  (tmp=%s)", tmp)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
