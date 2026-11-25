#!/bin/bash
#SBATCH -J cart_prep
#SBATCH -p quick,short
#SBATCH -c 8
#SBATCH --mem=32G
#SBATCH -t 01:30:00
#SBATCH -o /home/kle6/cart/logs/01_prep.out
#SBATCH -e /home/kle6/cart/logs/01_prep.out
# Walltime is deliberately short. A 4 h request for this stage sat at
# PENDING (Priority); the same job at 1.5 h started immediately, because short
# jobs backfill into gaps the scheduler cannot otherwise use. The stage itself
# takes under an hour.
set -euo pipefail
source /home/kle6/cart/scripts/slurm/env.sh
python scripts/01_prepare_data.py --n-passages "${N_PASSAGES:-2000}"
echo "PREP COMPLETE"
