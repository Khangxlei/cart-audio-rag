"""Transcribe the corpus under each noise condition, keeping word confidences
and beam-search alternatives.  Resumable: already-done doc_ids are skipped."""
import argparse, json, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cart.config import ASRConfig, DATA, RUNS
from cart.asr import NBestWhisper
from cart import audio as A
from cart.data import read_jsonl

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--conditions", nargs="+", default=["clean", "snr5", "snr0", "snr-5"])
    ap.add_argument("--model", default="small.en")
    ap.add_argument("--beam-size", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()

    out_dir = RUNS / "asr" / a.model.replace("/", "_")
    out_dir.mkdir(parents=True, exist_ok=True)
    corpus = read_jsonl(DATA / "corpus.jsonl")
    if a.limit:
        corpus = corpus[: a.limit]

    cfg = ASRConfig(model=a.model, beam_size=a.beam_size, num_hypotheses=a.beam_size)
    model = NBestWhisper(cfg)

    for cond in a.conditions:
        path = out_dir / f"{cond}.jsonl"
        done = {json.loads(l)["doc_id"] for l in open(path, encoding="utf-8")} if path.exists() else set()
        todo = [d for d in corpus if d["doc_id"] not in done]
        print(f"[{cond}] {len(done)} cached, {len(todo)} to go", flush=True)
        t0 = time.time()
        with open(path, "a", encoding="utf-8") as f:
            for i, d in enumerate(todo, 1):
                x = A.degrade(A.load(DATA.parent / d["audio"]), cond, seed=1)
                r = model.transcribe(x, d["doc_id"], cond)
                f.write(json.dumps(r.to_json(), ensure_ascii=False) + "\n")
                if i % 25 == 0:
                    f.flush()
                    el = time.time() - t0
                    print(f"  [{cond}] {i}/{len(todo)}  {el/i:.1f}s/doc  eta {(len(todo)-i)*el/i/60:.1f}min", flush=True)
        print(f"[{cond}] done in {(time.time()-t0)/60:.1f} min", flush=True)
