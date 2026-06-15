#!/bin/bash
#SBATCH --job-name=guidance_bench
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err
#
# Submit with positional args:
#   sbatch scripts/slurm_job.sh full_test flux2_klein_base geneval cfg pag sag
#   sbatch scripts/slurm_job.sh smoke_test flux2_klein_base all cfg
# Or with env vars:
#   sbatch --export=ALL,MODE=full_test,MODEL=flux2_klein_base,BENCHES="geneval dpg",METHODS="cfg pag sag" scripts/slurm_job.sh

set -euo pipefail

module purge
module load gnu14/14.1

PROJECT_ROOT=${PROJECT_ROOT:-$HOME/cfg_evaluation}

cd "$PROJECT_ROOT"

if [[ $# -gt 0 ]]; then
    MODE=$1
    shift
    case "$MODE" in
        smoke_test|evaluation|full_test)
            bash "scripts/${MODE}.sh" "$@"
            ;;
        *)
            bash scripts/full_test.sh "$MODE" "$@"
            ;;
    esac
else
    MODE=${MODE:-full_test}
    MODEL=${MODEL:?"MODEL env var required"}
    BENCHES=${BENCHES:?"BENCHES env var required, e.g. BENCHES='geneval dpg' or BENCHES=all"}
    METHODS=${METHODS:?"METHODS env var required, e.g. METHODS='cfg pag sag'"}
    # shellcheck disable=SC2086
    bash "scripts/${MODE}.sh" "$MODEL" $BENCHES $METHODS -- ${EXTRA_ARGS:-}
fi
