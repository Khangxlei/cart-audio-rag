#!/usr/bin/env bash
# One-time environment build on the Turing login node (which has internet).
set -euo pipefail
ROOT=/home/kle6/cart
module load miniconda3/25.1.1/24g7bpu
export CONDA_PKGS_DIRS=$ROOT/.conda_pkgs
export PIP_CACHE_DIR=$ROOT/.pip_cache
export HF_HOME=$ROOT/.hf_cache

if [ ! -d "$ROOT/env" ]; then
  conda create -y -p "$ROOT/env" python=3.11
fi
source activate "$ROOT/env"
python -m pip install --upgrade pip
python -m pip install -r "$ROOT/requirements.txt"
python -m nltk.downloader -d "$ROOT/.nltk_data" averaged_perceptron_tagger averaged_perceptron_tagger_eng cmudict
python - <<'PY'
import torch, faster_whisper, transformers, g2p_en, rank_bm25, jiwer, soundfile
print("torch", torch.__version__, "| transformers", transformers.__version__)
print("env OK")
PY
