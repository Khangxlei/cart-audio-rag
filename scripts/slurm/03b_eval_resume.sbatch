#!/bin/bash
#SBATCH -J cart_eval_resume
#SBATCH -p quick,short
#SBATCH --gres=gpu:1
#SBATCH --exclude=gpu-6-[01-20]
#SBATCH -c 8
#SBATCH --mem=32G
#SBATCH -t 08:00:00
#SBATCH -o /home/kle6/cart/logs/03b_eval_resume.out
#SBATCH -e /home/kle6/cart/logs/03b_eval_resume.out
# Resume the evaluation chain stage by stage.
#
# 03_eval.sbatch requests 3 h, the most that was promptly schedulable, and a
# running job cannot have its limit raised (that is admin-only). Each stage
# writes exactly one artifact, so this job re-runs only the stages whose
# artifact is missing, and is a near no-op if the first job finished
# everything. Submitted with --dependency=afterany so it runs whether the
# first job completed or hit the wall.
set -euo pipefail
source /home/kle6/cart/scripts/slurm/env.sh
M="${ASR_MODEL:-small.en}"
R=/home/kle6/cart/results

run_if_missing () {   # $1 = artifact, rest = command
  local art="$1"; shift
  if [ -s "$R/$art" ]; then
    echo "SKIP  $art present"
  else
    echo "RUN   $art missing -> $*"
    "$@"
  fi
}

run_if_missing tuning.json    python scripts/03_tune_cart.py      --model "$M"
run_if_missing retrieval.json python scripts/04_eval_retrieval.py --model "$M"
# 06 is self-resuming (skips conditions already in ablations.json), so always run it:
python scripts/06_ablations.py --model "$M" --conditions clean snr5 snr0 snr-5
# 05 is self-resuming (skips conditions already in qa.json), so always run it:
python scripts/05_eval_qa.py        --model "$M" --conditions clean snr0 snr-5 --max-queries 400 --judge
python scripts/07_report.py   # always regenerate from whatever exists
echo "ALL STAGES COMPLETE"
