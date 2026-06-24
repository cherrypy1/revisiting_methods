#!/usr/bin/env bash
# Wait for run_strat_remainder.sh to finish, then retry cfg dpg eval.
set -eu
cd ~/geneval
module purge
module load gnu14/14.1
source ~/.venv/bin/activate
TAG=strat27042026
LOG=~/geneval/outputs/logs/cfg_dpg_eval_retry_${TAG}.log
PID=222814
while kill -0 $PID 2>/dev/null; do sleep 60; done
echo "[$(date '+%F %T')] remainder done; starting cfg dpg eval retry" | tee -a $LOG
python scripts/eval_dpg.py --dpg-root /home/aaturevich/ELLA \
  --image-dir ~/geneval/outputs/dpg/cfg_strat27042026/images \
  --pic-num 1 --resolution 1024 \
  --out-dir ~/geneval/outputs/dpg/cfg_strat27042026/eval 2>&1 | tee -a $LOG
echo "[$(date '+%F %T')] cfg dpg eval retry done" | tee -a $LOG
