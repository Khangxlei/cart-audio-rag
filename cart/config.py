"""Central configuration for the CART pipeline."""
from __future__ import annotations

import os
from dataclasses import dataclass, field, asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# Keep model/dataset downloads on the project drive: the default cache lives on
# the system drive, which this project would otherwise fill with audio shards.
os.environ.setdefault("HF_HOME", str(ROOT / ".hf_cache"))

DATA = ROOT / "data"
RUNS = ROOT / "runs"
RESULTS = ROOT / "results"
for _d in (DATA, RUNS, RESULTS):
    _d.mkdir(exist_ok=True)


@dataclass
class ASRConfig:
    model: str = "small.en"          # faster-whisper checkpoint
    device: str = "cuda"
    compute_type: str = "int8_float16"
    beam_size: int = 5               # K beams == K hypotheses we keep
    num_hypotheses: int = 5
    language: str = "en"
    word_timestamps: bool = True
    condition_on_previous_text: bool = False
    # faster-whisper falls back to sampling at temperature>0 which destroys the
    # beam structure we need, so we keep a single greedy/beam temperature.
    temperature: tuple = (0.0,)


@dataclass
class CARTConfig:
    tau: float = 0.60        # word-confidence threshold for "low confidence"
    lam: float = 1.0         # lambda in s = s_text + lam * (1 - c) * s_phonetic
    alpha: float = 0.5       # weight of alternative-spelling postings in BM25 tf
    theta: float = 0.60      # minimum phonetic similarity to count as a match
    max_alts: int = 3        # max alternative spellings kept per low-conf word
    bm25_k1: float = 0.9
    bm25_b: float = 0.4
    use_alt_index: bool = True
    use_phonetic: bool = True
    confidence_gate: bool = True   # if False, drop the (1 - c) factor


@dataclass
class ExperimentConfig:
    n_passages: int = 400
    max_questions_per_passage: int = 100
    seed: int = 13
    noise_conditions: tuple = ("clean", "snr10", "snr5", "snr0")
    dev_fraction: float = 0.25     # queries used to tune lambda/tau
    top_k_context: int = 3         # passages given to the reader LLM
    reader_model: str = "Qwen/Qwen2.5-1.5B-Instruct"
    dense_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    asr: ASRConfig = field(default_factory=ASRConfig)
    cart: CARTConfig = field(default_factory=CARTConfig)

    def to_dict(self):
        return asdict(self)


DEFAULT = ExperimentConfig()
