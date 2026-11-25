#!/bin/bash
#SBATCH -J cart_gputest
#SBATCH -p quick,short
#SBATCH --gres=gpu:1
#SBATCH -c 4
#SBATCH --mem=16G
#SBATCH -t 00:10:00
#SBATCH -o /home/kle6/cart/logs/gputest.out
#SBATCH -e /home/kle6/cart/logs/gputest.out
set -euo pipefail
source /home/kle6/cart/scripts/slurm/env.sh
python scripts/validate_env.py
