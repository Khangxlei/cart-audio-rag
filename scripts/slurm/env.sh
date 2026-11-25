# Shared job preamble.
ROOT=/home/kle6/cart
cd "$ROOT"
source /etc/profile.d/modules.sh
module load miniconda3/25.1.1/24g7bpu
source activate "$ROOT/env"
export HF_HOME="$ROOT/.hf_cache"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"      # assets are prefetched on the login node
export NLTK_DATA="/home/kle6/nltk_data:$ROOT/.nltk_data"
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

# CTranslate2 links against cuDNN 9 and cuBLAS at runtime. Those libraries ship
# inside the pip nvidia-* wheels that torch pulls in, but nothing puts them on
# the loader path, so faster-whisper aborts with
#   Unable to load any of {libcudnn_ops.so.9.1.0, libcudnn_ops.so.9, ...}
# unless they are added explicitly.
SITE="$(python -c 'import site; print(site.getsitepackages()[0])')"
export LD_LIBRARY_PATH="$SITE/nvidia/cudnn/lib:$SITE/nvidia/cublas/lib:${LD_LIBRARY_PATH:-}"

nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || true
