#!/usr/bin/env bash
#SBATCH --job-name=strat
#SBATCH --partition=normal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=12:00:00
#SBATCH --output=/home/aaturevich/geneval/logs/strat_%j.out
#SBATCH --error=/home/aaturevich/geneval/logs/strat_%j.out

set -euo pipefail
mkdir -p /home/aaturevich/geneval/logs
cd /home/aaturevich/geneval
exec bash scripts/run_strat.sh
