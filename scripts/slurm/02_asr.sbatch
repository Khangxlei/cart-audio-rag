#!/bin/bash
#SBATCH -J cart_asr
#SBATCH -p quick,short
#SBATCH --gres=gpu:1
#SBATCH --exclude=gpu-6-[01-20]
#SBATCH -c 8
#SBATCH --mem=48G
#SBATCH -t 02:00:00
#SBATCH -a 0-3
#SBATCH -o /home/kle6/cart/logs/02_asr_%a.out
#SBATCH -e /home/kle6/cart/logs/02_asr_%a.out
# One array task per acoustic condition. Each took 33-42 min on an L40S or H100.
#
# --exclude: gpu-6-[01-20] are RTX PRO 6000 Blackwell (sm_120), which CUDA 12.1
# predates. CTranslate2 dies there with CUBLAS_STATUS_NOT_SUPPORTED, and torch
# would too. Asking for a generic gpu:1 rather than a specific model widens the
# pool of nodes that can schedule the job.
#
# This stage is resumable: it skips documents already present in the output
# JSONL, so a preempted or node-failed task loses time, not work.
set -euo pipefail
source /home/kle6/cart/scripts/slurm/env.sh
CONDS=(clean snr5 snr0 snr-5)
COND=${CONDS[$SLURM_ARRAY_TASK_ID]}
echo "condition=$COND model=${ASR_MODEL:-small.en}"
python scripts/02_run_asr.py --conditions "$COND" --model "${ASR_MODEL:-small.en}" --beam-size 8
echo "ASR COMPLETE $COND"
