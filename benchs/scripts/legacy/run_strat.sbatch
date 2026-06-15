#!/usr/bin/env bash
#SBATCH --job-name=strat
#SBATCH --partition=normal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=12:00:00
# NOTE: #SBATCH directives can't expand $HOME — paths are relative to the submit
# dir, so submit this from the project root (~/cfg_evaluation).
#SBATCH --output=logs/strat_%j.out
#SBATCH --error=logs/strat_%j.out

set -euo pipefail
cd "$HOME/cfg_evaluation"
mkdir -p logs
exec bash scripts/run_strat.sh
