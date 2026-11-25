"""Download every remote asset on the login node so the compute jobs can run
offline. Pure network I/O - no heavy compute on the login node."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cart.config import DEFAULT          # sets HF_HOME before anything imports hub
from cart.data import HF_REPO, N_SHARDS, load_squad_dev

from huggingface_hub import hf_hub_download, snapshot_download

print("squad dev ...", flush=True)
load_squad_dev()

print("spoken-squad shards ...", flush=True)
for s in range(N_SHARDS):
    hf_hub_download(HF_REPO, f"data/test-{s:05d}-of-{N_SHARDS:05d}.parquet", repo_type="dataset")
    print(f"  shard {s+1}/{N_SHARDS}", flush=True)

for repo in [f"Systran/faster-whisper-{m}" for m in ("small.en", "large-v3")] + \
            [DEFAULT.reader_model, DEFAULT.dense_model]:
    print("model", repo, "...", flush=True)
    snapshot_download(repo)

print("PREFETCH COMPLETE", flush=True)
