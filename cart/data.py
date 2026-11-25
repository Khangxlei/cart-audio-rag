"""Dataset preparation for Spoken-SQuAD.

`AudioLLMs/spoken_squad_test` ships the Spoken-SQuAD test split as
(audio_context, question, answer) triples, with the *same* passage audio
repeated once per question and no passage identifier.  We therefore

  1. de-duplicate the audio by content hash to recover the passage corpus,
  2. re-attach the gold reference transcript by matching the question string
     against SQuAD v1.1 dev (Spoken-SQuAD's test split is built from it),

which gives us a proper ad-hoc retrieval collection: ~N passages, each with a
gold transcript, plus a set of questions whose single relevant passage is known.
"""
from __future__ import annotations

import hashlib
import io
import json
import urllib.request
from collections import defaultdict
from pathlib import Path

import numpy as np
import soundfile as sf

from .config import DATA

SQUAD_DEV_URL = "https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v1.1.json"
HF_REPO = "AudioLLMs/spoken_squad_test"
N_SHARDS = 21


def _norm_q(q: str) -> str:
    return " ".join(q.lower().split()).strip(" ?.!")


def load_squad_dev(cache: Path = DATA / "dev-v1.1.json") -> dict:
    """question -> {context, title, answers}."""
    if not cache.exists():
        cache.write_bytes(urllib.request.urlopen(SQUAD_DEV_URL, timeout=120).read())
    raw = json.loads(cache.read_text(encoding="utf-8"))
    lut = {}
    for art in raw["data"]:
        for para in art["paragraphs"]:
            for qa in para["qas"]:
                lut[_norm_q(qa["question"])] = {
                    "context": para["context"],
                    "title": art["title"],
                    "answers": sorted({a["text"] for a in qa["answers"]}),
                }
    return lut


def build_corpus(n_passages: int, seed: int = 13, max_shards: int = N_SHARDS) -> tuple[list, list]:
    """Download shards until `n_passages` distinct passages are collected."""
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download

    squad = load_squad_dev()
    audio_dir = DATA / "audio"
    audio_dir.mkdir(exist_ok=True)

    by_hash: dict[str, dict] = {}
    rows_by_hash: dict[str, list] = defaultdict(list)

    for shard in range(max_shards):
        if len(by_hash) >= n_passages:
            break
        path = hf_hub_download(
            HF_REPO, f"data/test-{shard:05d}-of-{N_SHARDS:05d}.parquet", repo_type="dataset"
        )
        for row in pq.read_table(path).to_pylist():
            raw = row["context"]["bytes"]
            h = hashlib.md5(raw).hexdigest()
            if h not in by_hash and len(by_hash) >= n_passages:
                continue          # corpus is full; ignore unseen passages
            if h not in by_hash:
                by_hash[h] = {"bytes": raw}
            rows_by_hash[h].append(row)

    # resolve gold transcripts and write audio
    corpus, queries = [], []
    seen_qids: set[str] = set()
    for i, (h, blob) in enumerate(sorted(by_hash.items())):
        rows = rows_by_hash[h]
        hits = [squad[_norm_q(r["instruction"])] for r in rows if _norm_q(r["instruction"]) in squad]
        if not hits:
            continue
        # the modal context among matched questions is the passage transcript
        ctx = max({hh["context"] for hh in hits}, key=lambda c: sum(hh["context"] == c for hh in hits))
        title = next(hh["title"] for hh in hits if hh["context"] == ctx)

        # content-addressed so that growing the corpus never renumbers the
        # passages already transcribed (ASR caches stay valid)
        doc_id = f"d_{h[:12]}"
        wav = audio_dir / f"{doc_id}.wav"
        if not wav.exists():
            data, sr = sf.read(io.BytesIO(blob["bytes"]))
            sf.write(wav, data, sr)
        else:
            data, sr = sf.read(wav)
        corpus.append(
            {
                "doc_id": doc_id,
                "md5": h,
                "title": title,
                "gold_text": ctx,
                "audio": str(wav.relative_to(DATA.parent)),
                "duration": round(len(data) / sr, 2),
            }
        )
        for r in rows:
            key = _norm_q(r["instruction"])
            if key not in squad or squad[key]["context"] != ctx:
                continue
            qid = "q_" + hashlib.md5(f"{doc_id}|{key}".encode()).hexdigest()[:12]
            if qid in seen_qids:
                continue
            seen_qids.add(qid)
            queries.append(
                {
                    "qid": qid,
                    "question": r["instruction"],
                    "answers": sorted({r["answer"], *squad[key]["answers"]}),
                    "gold_doc_id": doc_id,
                }
            )

    # Deterministic dev/test split, hashed off the qid so that it does not shift
    # when the corpus grows (dev is only ever used to tune CART's hyperparameters).
    for q in queries:
        bucket = int(hashlib.md5((q["qid"] + f"|{seed}").encode()).hexdigest(), 16) % 100
        q["split"] = "dev" if bucket < 25 else "test"
    return corpus, queries


def write_jsonl(path: Path, rows) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def read_jsonl(path: Path) -> list:
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]
