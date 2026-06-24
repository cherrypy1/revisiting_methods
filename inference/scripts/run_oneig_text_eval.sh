#!/usr/bin/env bash
# Fast OneIG text-only validation over already generated images.
#
# Intended to run on the cluster from the inference repo. It does not generate
# images: it creates manual eval shards from existing *_oneig outputs and
# submits one worker job per shard.

set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$ROOT"

PROMPTS=${ONEIG_TEXT_PROMPTS:-all}
PARTITION=${ONEIG_TEXT_PARTITION:-rocky}
GPU_TYPE=${ONEIG_TEXT_GPU_TYPE:-v100}
CPUS=${ONEIG_TEXT_CPUS:-6}
WALLTIME=${ONEIG_TEXT_WALLTIME:-08:00:00}
PYTHON_BIN=${ONEIG_TEXT_PYTHON:-$HOME/_.venv/.venv/bin/python}
if [[ "$PROMPTS" == "all" || "$PROMPTS" == "ALL" || "$PROMPTS" == "full" || "$PROMPTS" == "FULL" ]]; then
    PROMPT_SUFFIX=text_all
else
    PROMPT_SUFFIX=text${PROMPTS}
fi
SHARD_DIR=${ONEIG_TEXT_SHARD_DIR:-outputs/_manual_shards/${PROMPT_SUFFIX}}
HF_HOME=${HF_HOME:-$HOME/.cache/huggingface}

echo "== OneIG text-only eval =="
echo "root:       $ROOT"
echo "prompts:    $PROMPTS"
echo "partition:  $PARTITION"
echo "gpu:        $GPU_TYPE"
echo "walltime:   $WALLTIME"
echo "python:     $PYTHON_BIN"
echo "shard dir:  $SHARD_DIR"
echo "HF_HOME:    $HF_HOME"

if [[ ! -x "$PYTHON_BIN" ]]; then
    echo "Python executable not found or not executable: $PYTHON_BIN" >&2
    exit 1
fi

echo
echo "== Remove stale OneIG text eval scratch only =="
find outputs -maxdepth 3 -type d -path "*/_evalwork/*_oneig_text_eval" -print
find outputs -maxdepth 3 -type d -path "*/_evalwork/*_oneig_text_eval" -exec rm -rf {} +

echo
echo "== Build manual text shards =="
mkdir -p "$SHARD_DIR"
"$PYTHON_BIN" - "$PROMPTS" "$SHARD_DIR" <<'PY'
import json
import sys
from pathlib import Path

raw_limit = sys.argv[1].strip().lower()
limit = None if raw_limit in {"all", "full"} else int(raw_limit)
suffix = "text_all" if limit is None else f"text{limit}"
shard_dir = Path(sys.argv[2])
base = Path("outputs")
shard_dir.mkdir(parents=True, exist_ok=True)

written = 0
for campaign in sorted(base.glob("*_oneig")):
    manifest = campaign / "manifest.jsonl"
    if not manifest.is_file():
        continue

    item_ids = []
    first = None
    for line in manifest.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if (
            row.get("bench") != "oneig"
            or row.get("category") != "text"
            or row.get("sample_idx") != 0
        ):
            continue

        item_id = row["item_id"]
        model, config, bench, category, prompt_id, sample = item_id.split("/")
        sample_idx = sample[1:]
        image = campaign / model / config / bench / category / prompt_id / f"sample_{sample_idx}.png"
        sidecar = campaign / model / config / bench / category / prompt_id / f"sample_{sample_idx}.json"
        if image.is_file() and sidecar.is_file():
            item_ids.append(item_id)
            first = first or (model, config)

    if limit is not None:
        item_ids = item_ids[:limit]
        if len(item_ids) < limit:
            print(f"SKIP {campaign.name}: only {len(item_ids)} generated text items")
            continue
    elif not item_ids:
        print(f"SKIP {campaign.name}: no generated text items")
        continue

    model, config = first
    shard = {
        "model": model,
        "config": config,
        "bench": "oneig",
        "phase": "eval",
        "item_ids": item_ids,
        "campaign_out": str(campaign.resolve()),
        "category": "text",
        "seed": 0,
        "verbose": False,
    }
    out = shard_dir / f"{campaign.name}_{suffix}_eval.json"
    out.write_text(json.dumps(shard, ensure_ascii=False, indent=2))
    print(f"WROTE {out}: {len(item_ids)} items")
    written += 1

if written == 0:
    raise SystemExit("no text eval shards were written")
PY

echo
echo "== Submit text eval workers =="
for shard in "$SHARD_DIR"/*_"${PROMPT_SUFFIX}"_eval.json; do
    [[ -e "$shard" ]] || continue
    name=$(basename "$shard" .json)
    job_script="$SHARD_DIR/${name}.sbatch"
    cat > "$job_script" <<EOF
#!/usr/bin/env bash
#SBATCH -p $PARTITION
#SBATCH --gres=gpu:${GPU_TYPE}:1
#SBATCH -c $CPUS
#SBATCH -t $WALLTIME
#SBATCH -J $name
#SBATCH -o $SHARD_DIR/${name}.%j.out

set -euo pipefail
source /etc/profile.d/modules.sh 2>/dev/null || true
module purge 2>/dev/null || true
module load gnu14/14.1
GXX_LIB=\$(g++ -print-file-name=libstdc++.so.6 2>/dev/null || true)
if [ -n "\$GXX_LIB" ] && [ -f "\$GXX_LIB" ]; then
    export LD_LIBRARY_PATH="\$(dirname "\$GXX_LIB"):\${LD_LIBRARY_PATH:-}"
fi
source "$HOME/_.venv/.venv/bin/activate"
cd "$ROOT"
export TMPDIR=/tmp/\$USER/oneig_text${PROMPTS}_\$SLURM_JOB_ID
mkdir -p "\$TMPDIR"
unset ONEIG_HF_HOME CFG_HF_HOME
export HF_HOME=$HF_HOME
mkdir -p "\$HF_HOME"
export HF_HUB_DISABLE_XET=1
"$PYTHON_BIN" -m cfgbench.cli worker --shard "$shard"
EOF
    sbatch "$job_script"
done

echo
echo "== Active jobs =="
squeue -u "$USER" || true

echo
echo "Logs: $SHARD_DIR/*.out"
echo "Summaries after jobs finish:"
echo "  for d in outputs/*_oneig; do echo \"### \$d\"; cat \"\$d/flux2_klein_base\"/*/oneig/text/_summary.json 2>/dev/null || true; done"
