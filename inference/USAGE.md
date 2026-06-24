# cfgbench — usage

Generic, autonomous, resumable benchmark runner. Adapter contract: ADAPTERS.md.
Lives in `cfgbench/`, reuses `pipelines/` + `configs/` unchanged, and can run from any repo path.

## Setup

- Server run env: `module load gnu14/14.1 && source ~/_.venv/.venv/bin/activate` (the orchestrator's srun
  steps do this automatically via `STEP_PRE`).
- Telegram alerts (optional): create a bot via @BotFather, then
  ```
  mkdir -p ~/.config/cfgbench
  printf 'export TELEGRAM_BOT_TOKEN=...\nexport TELEGRAM_CHAT_ID=...\n' > ~/.config/cfgbench/notify.env
  chmod 600 ~/.config/cfgbench/notify.env
  ```
  Verify: `python -m cfgbench.cli notify-test`.

## Run a campaign (autonomous)

Edit a spec under `configs/campaigns/` (see `main.yaml` full, `smoke.yaml` tiny,
`flux_smoke.yaml` for FLUX.2-klein). Then on the login node:

```
cd ~/inference
module load gnu14/14.1
source ~/_.venv/.venv/bin/activate
mkdir -p outputs

nohup ~/_.venv/.venv/bin/python -m cfgbench.cli run configs/campaigns/main.yaml \
  > outputs/main.run.log 2>&1 &
```

FLUX smoke:

```
nohup ~/_.venv/.venv/bin/python -m cfgbench.cli run configs/campaigns/flux_smoke.yaml \
  > outputs/flux_smoke.run.log 2>&1 &
```

## FLUX half-subset campaigns

These campaign specs are meant for overnight FLUX.2-klein runs on the cluster:

- `no_cfg_geneval`, `cfg_geneval`, `pag_geneval`, `seg_geneval`, `cfgpp_geneval`, `oseg_geneval`,
  `tcfg_geneval`, `cfg0s_geneval`, `apg_geneval`, `sag_geneval`: half of GenEval, about 277 images each.
- `no_cfg_dpg`, `pag_dpg`, `seg_dpg`, `cfgpp_dpg`, `oseg_dpg`, `tcfg_dpg`, `cfg0s_dpg`, `apg_dpg`:
  half of DPG, about 533 images each.
- `cfg_oneig`, `pag_oneig`, `cfgpp_oneig`, `seg_oneig`: half of OneIG, about 316 images each.

Start from the cluster login node:

```bash
cd ~/inference
module load gnu14/14.1
source ~/_.venv/.venv/bin/activate
mkdir -p outputs
```

GenEval generation can run in parallel:

```bash
for c in no_cfg_geneval cfg_geneval pag_geneval seg_geneval cfgpp_geneval oseg_geneval tcfg_geneval cfg0s_geneval apg_geneval sag_geneval; do
  nohup ~/_.venv/.venv/bin/python -m cfgbench.cli run configs/campaigns/${c}.yaml --stage generate \
    > outputs/${c}.run.log 2>&1 &
done
```

OneIG uses Qwen2.5-VL scorers. Patch the external OneIG checkout once on V100 nodes so it
does not force FlashAttention2:

```bash
~/_.venv/.venv/bin/python scripts/patch_oneig_eval.py
```

OneIG generation can run in parallel without touching the shared OneIG scorer outputs:

```bash
for c in no_cfg_oneig cfg_oneig pag_oneig seg_oneig cfgpp_oneig oseg_oneig tcfg_oneig cfg0s_oneig apg_oneig; do
  nohup ~/_.venv/.venv/bin/python -m cfgbench.cli run configs/campaigns/${c}.yaml --stage generate \
    > outputs/${c}.run.log 2>&1 &
done
```

OneIG validation writes through `$ONEIG_ROOT/results`, so cfgbench serializes eval with a lock.
For easier monitoring, validate one campaign at a time:

```bash
for c in no_cfg_oneig cfg_oneig pag_oneig seg_oneig cfgpp_oneig oseg_oneig tcfg_oneig cfg0s_oneig apg_oneig; do
  nohup ~/_.venv/.venv/bin/python -m cfgbench.cli run configs/campaigns/${c}.yaml --stage validate \
    > outputs/${c}.validate.log 2>&1
done
```

DPG validation needs mPLUG available locally. If `~/.config/cfgbench/dpg.env` does not exist yet,
download mPLUG once and write the env file:

```bash
mkdir -p ~/.config/cfgbench

python - <<'PY'
from pathlib import Path

try:
    from modelscope import snapshot_download
except ImportError:
    from modelscope.hub.snapshot_download import snapshot_download

path = snapshot_download(
    "damo/mplug_visual-question-answering_coco_large_en",
    cache_dir=str(Path.home() / ".cache/modelscope"),
)

env = Path.home() / ".config/cfgbench/dpg.env"
env.write_text(f'export MPLUG_VQA_MODEL="{path}"\n')
print(path)
print("wrote", env)
PY
```

Patch ELLA once so DPG eval reads `MPLUG_VQA_MODEL` instead of checking ModelScope online:

```bash
python - <<'PY'
from pathlib import Path

p = Path.home() / "ELLA/dpg_bench/compute_dpg_bench.py"
s = p.read_text()
needle = "damo/mplug_visual-question-answering_coco_large_en"

if "MPLUG_VQA_MODEL" not in s:
    if needle not in s:
        raise SystemExit(f"cannot find {needle} in {p}")
    if "import os" not in s:
        s = s.replace("import argparse\n", "import argparse\nimport os\n", 1)
    repl = f'os.environ.get("MPLUG_VQA_MODEL", "{needle}")'
    s = s.replace(f'"{needle}"', repl)
    s = s.replace(f"'{needle}'", repl)
    p.write_text(s)

print("patched", p)
PY

grep -n "MPLUG_VQA_MODEL\|mplug_visual" ~/ELLA/dpg_bench/compute_dpg_bench.py
```

Run DPG campaigns with separate `accelerate` ports:

```bash
source ~/.config/cfgbench/dpg.env
test -d "$MPLUG_VQA_MODEL" && echo "mPLUG OK: $MPLUG_VQA_MODEL"

i=0
for c in no_cfg_dpg pag_dpg seg_dpg cfgpp_dpg oseg_dpg tcfg_dpg cfg0s_dpg apg_dpg; do
  port=$((29500 + i))
  nohup env MPLUG_VQA_MODEL="$MPLUG_VQA_MODEL" DPG_PORT="$port" \
    ~/_.venv/.venv/bin/python -m cfgbench.cli run configs/campaigns/${c}.yaml \
    > outputs/${c}.run.log 2>&1 &
  i=$((i + 1))
done
```

The runner is resumable: rerunning the same campaign skips existing generated images and continues
with unfinished generation/eval work.

Monitor all half-subset campaigns:

```bash
for c in no_cfg_dpg pag_dpg seg_dpg cfgpp_dpg oseg_dpg tcfg_dpg cfg0s_dpg apg_dpg \
         no_cfg_geneval cfg_geneval pag_geneval seg_geneval cfgpp_geneval oseg_geneval tcfg_geneval cfg0s_geneval apg_geneval sag_geneval \
         no_cfg_oneig cfg_oneig pag_oneig seg_oneig cfgpp_oneig oseg_oneig tcfg_oneig cfg0s_oneig apg_oneig; do
  ~/_.venv/.venv/bin/python -m cfgbench.cli status configs/campaigns/${c}.yaml
done
```

The orchestrator auto-allocates `pool.max_jobs` 1-GPU jobs on `rocky` (sbatch holders), dispatches
gen then eval shards, replaces dead jobs, and releases jobs when done. It's resumable: rerun the same
command and it continues from the filesystem (skips finished items, reattaches live jobs via `pool.json`).
Run it under `nohup`/`tmux` so it survives disconnect.

Scope knobs in the spec:
- `limit: {geneval: 60, oneig: 30, dpg: 90}` — stratified subset per bench (omit → full sets).
- `samples_per_prompt: {geneval: 4, oneig: 1, dpg: 1}`.
- `pool.max_jobs` — parallel 1-GPU jobs; `pool.walltime`; `pool.gpu_type` (v100 default).
- `sample_seed` / `seed` — reproducible subset / generation.

## Monitor

```
python -m cfgbench.cli status  configs/campaigns/main.yaml   # progress + live worker throughput
python -m cfgbench.cli events  configs/campaigns/main.yaml -n 30   # lifecycle timeline
python -m cfgbench.cli health  configs/campaigns/main.yaml   # exit!=0 on anomaly (cron-friendly)
```

Health checks (in-loop + the `health` command): pool starvation, stall, low `$HOME` disk, failure spike.
On CRITICAL it writes `outputs/<name>/ALERT`, logs an event, and (if configured) sends Telegram.
Optional cron escalation: `*/15 * * * * cd /path/to/inference && python -m cfgbench.cli health configs/campaigns/main.yaml || <ping>`.

## Results

```
outputs/<name>/<model>/<config>/<bench>/<category>/<prompt_id>/sample_<i>.png + sample_<i>.json
                                          .../_summary.json   (category / bench / run levels)
outputs/<name>/{manifest.jsonl, events.jsonl, pool.json, status.json, logs/}
```
Each `sample_<i>.json` carries the prompt, params, gen time, and metrics. Build tables:

```
python -m cfgbench.cli report configs/campaigns/main.yaml    # writes report.md + report.csv
```

## Add a model or benchmark

One adapter file + one registry line — see ADAPTERS.md. Core (queue/SLURM/layout/logging) is untouched.
