"""Build the Spoken-SQuAD retrieval collection (corpus + queries + audio)."""
import argparse, json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cart.config import DATA
from cart.data import build_corpus, write_jsonl

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-passages", type=int, default=400)
    ap.add_argument("--seed", type=int, default=13)
    a = ap.parse_args()

    corpus, queries = build_corpus(a.n_passages, a.seed)
    write_jsonl(DATA / "corpus.jsonl", corpus)
    write_jsonl(DATA / "queries.jsonl", queries)
    dur = sum(c["duration"] for c in corpus)
    print(json.dumps({
        "passages": len(corpus),
        "queries": len(queries),
        "dev": sum(q["split"] == "dev" for q in queries),
        "test": sum(q["split"] == "test" for q in queries),
        "audio_hours": round(dur / 3600, 2),
        "mean_passage_words": round(sum(len(c["gold_text"].split()) for c in corpus) / len(corpus), 1),
    }, indent=2))
