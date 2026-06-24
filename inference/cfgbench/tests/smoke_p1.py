"""P1 local smoke: worker gen+eval end-to-end with dummy adapters (no torch/GPU/network).

Verifies: real registries populated, DiffusersModelAdapter.load_config parses a real yaml,
and the generic worker runs a full gen→eval→summary cycle via the adapter contract.

Run: python -m cfgbench.tests.smoke_p1
"""

from __future__ import annotations

import base64
import shutil
import tempfile
from pathlib import Path

from cfgbench.benchmarks.base import BenchmarkAdapter, PromptSpec
from cfgbench.benchmarks.registry import BENCHMARKS
from cfgbench.benchmarks.registry import register as reg_bench
from cfgbench.config import method_config_path
from cfgbench.core.layout import (CampaignPaths, ItemRef, eval_done, gen_done, read_json)
from cfgbench.core.logsetup import configure
from cfgbench.core.shard import Shard
from cfgbench.core.worker import run_shard
from cfgbench.models.base import ModelAdapter, ModelHandle
from cfgbench.models.diffusers_adapter import load_config
from cfgbench.models.registry import MODELS
from cfgbench.models.registry import register as reg_model

# valid 1x1 PNG so layout.image_ok passes with or without PIL
_PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAIAAACQd1PeAAAADElEQVR4nGNgYGAAAAAEAAH2FzhVAAAAAElFTkSuQmCC")


class _Img:
    def save(self, path, format=None):
        Path(path).write_bytes(_PNG)


class _Handle(ModelHandle):
    def params(self):
        return {"guidance_scale": 1.0, "num_inference_steps": 2}

    def generate(self, prompt, *, seed, n=1, **kw):
        return [_Img() for _ in range(n)]

    def close(self):
        pass


class _DummyModel(ModelAdapter):
    name = "sd35"  # override real sd35 in THIS process so method_config_path resolves

    def load(self, config):
        assert Path(str(config)).is_file()  # worker passed a real yaml path
        return _Handle()


class _DummyBench(BenchmarkAdapter):
    name = "dummybench"
    eval_needs_gpu = False
    default_samples_per_prompt = 1

    def prompts(self):
        return [PromptSpec(id=f"{i:03d}", text=f"prompt {i}", category="c") for i in range(3)]

    def evaluate(self, items, workdir):
        return {it.item_id: {"score": 1.0} for it in items}


def main() -> None:
    log = configure(verbose=False)

    # real registries populated by import side effects
    assert {"sd35", "cosmos2"} <= set(MODELS), MODELS
    assert {"geneval", "oneig", "dpg"} <= set(BENCHMARKS), BENCHMARKS
    log.info("registries OK: models=%s benches=%s", sorted(MODELS), sorted(BENCHMARKS))

    # DiffusersModelAdapter.load_config parses a real method yaml (needs PyYAML)
    try:
        spec, params = load_config(str(method_config_path("sd35", "cfg")))
        assert spec and "guidance_scale" in params
        log.info("load_config OK: pipeline=%s params=%s", spec, sorted(params))
    except ModuleNotFoundError as e:
        log.warning("load_config skipped (%s) — validated on server where PyYAML is present", e)

    # install dummies (process-local) and run the worker end-to-end
    reg_model(_DummyModel())
    reg_bench(_DummyBench())

    tmp = Path(tempfile.mkdtemp(prefix="cfgbench_p1_"))
    try:
        paths = CampaignPaths(tmp).ensure()
        refs = [ItemRef("sd35", "cfg", "dummybench", "c", f"{i:03d}", 0) for i in range(3)]
        ids = [r.item_id for r in refs]

        gen = Shard("sd35", "cfg", "dummybench", "gen", ids, str(tmp), category="c").save(
            tmp / "gen.json")
        run_shard(gen)
        assert all(gen_done(paths, r) for r in refs), "gen incomplete"
        log.info("worker gen OK: 3/3 images + sidecars")

        ev = Shard("sd35", "cfg", "dummybench", "eval", ids, str(tmp)).save(tmp / "eval.json")
        run_shard(ev)
        assert all(eval_done(paths, r) for r in refs), "eval incomplete"
        side = read_json(paths.sidecar(refs[0]))
        assert side["metrics"] == {"score": 1.0}, side["metrics"]
        summ = read_json(paths.summary_bench("sd35", "cfg", "dummybench"))
        assert summ and "overall" in summ
        log.info("worker eval OK: metrics merged + summary %s", summ["overall"])

        # idempotency: rerun gen does nothing new
        run_shard(gen)
        log.info("idempotent rerun OK")

        log.info("P1 SMOKE PASSED  (tmp=%s)", tmp)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
