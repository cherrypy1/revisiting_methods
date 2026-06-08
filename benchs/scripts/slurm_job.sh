#!/bin/bash
#SBATCH --job-name=sd35_bench
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err
#
# Submit with e.g.:
#   sbatch --export=ALL,METHOD=seg_sigma10_cfg3,BENCH=geneval scripts/slurm_job.sh
# Or run multiple methods on a single benchmark:
#   sbatch --export=ALL,BENCH=geneval scripts/slurm_job.sh all

module purge
module load gnu14/14.1

PROJECT_ROOT=${PROJECT_ROOT:-$HOME/cfg_evaluation}
BENCH=${BENCH:?"BENCH env var required (geneval|oneig|dpg)"}

cd "$PROJECT_ROOT"

if [[ "${METHOD:-all}" == "all" ]]; then
    bash scripts/run_all_methods.sh "$BENCH"
else
    bash scripts/bench.sh "$METHOD" "$BENCH"
fi
