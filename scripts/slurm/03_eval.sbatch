#!/bin/bash
#SBATCH -J cart_eval
#SBATCH -p quick,short
#SBATCH --gres=gpu:1
#SBATCH --exclude=gpu-6-[01-20]
#SBATCH -c 8
#SBATCH --mem=32G
#SBATCH -t 03:00:00
#SBATCH -o /home/kle6/cart/logs/03_eval.out
#SBATCH -e /home/kle6/cart/logs/03_eval.out
# The whole evaluation chain in one job. It does NOT fit in 3 h on a cold start
# (tuning alone is ~80 min, the ablations ~2 h), and a running job's limit
# cannot be raised by a user -- only lowered. So the intended pattern is to
# submit this and then chain 03b_eval_resume.sbatch behind it with
# --dependency=afterany; every stage is self-resuming, so the chain picks up
# wherever the wall clock cut the previous link off.
set -euo pipefail
source /home/kle6/cart/scripts/slurm/env.sh
M="${ASR_MODEL:-small.en}"
python scripts/03_tune_cart.py      --model "$M"
python scripts/04_eval_retrieval.py --model "$M"
python scripts/06_ablations.py      --model "$M" --conditions clean snr5 snr0 snr-5
python scripts/05_eval_qa.py        --model "$M" --conditions clean snr0 snr-5 --max-queries 400 --judge
python scripts/07_report.py
echo "ALL STAGES COMPLETE"
