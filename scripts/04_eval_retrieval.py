"""Main retrieval experiment: CART vs. baselines across ASR degradation levels.

Systems
  oracle_bm25     BM25 on the human reference transcript        (upper bound)
  bm25_1best      BM25 on Whisper 1-best                        (baseline 1)
  nbest_uniform   BM25 + every beam hypothesis, untargeted      (baseline 2)
  dense_1best     neural dense retrieval on 1-best              (baseline 3*)
  cart            confidence-targeted expansion + phonetic fallback

  * stands in for SpeechRAG, whose speech encoder is not publicly available.

Reported on the held-out test split with 95% bootstrap CIs on MRR and paired
bootstrap p-values against bm25_1best.  Also reported on the subset of queries
that contain a named entity, where ASR errors concentrate.
"""
import argparse, json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from cart.config import CARTConfig, DATA, RESULTS
from cart.data import read_jsonl
from cart.diagnostics import index_cost, named_entity_tokens, transcript_report
from cart.evaluate import build_systems, evaluate_systems, filter_queries, load_condition
from cart.index import content_terms
from cart.asr import tokenize
from cart.phonetics import G2P


def ne_query_mask(queries, corpus):
    """Queries whose content terms include a named entity of the gold passage."""
    nes = {c["doc_id"]: named_entity_tokens(c["gold_text"]) for c in corpus}
    out = []
    for q in queries:
        terms = set(content_terms(tokenize(q["question"])))
        out.append(bool(terms & nes.get(q["gold_doc_id"], set())))
    return np.array(out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="small.en")
    ap.add_argument("--conditions", nargs="+", default=["clean", "snr5", "snr0", "snr-5"])
    ap.add_argument("--split", default="test")
    ap.add_argument("--config", default=str(RESULTS / "tuning.json"))
    ap.add_argument("--no-dense", action="store_true")
    ap.add_argument("--out", default=str(RESULTS / "retrieval.json"))
    a = ap.parse_args()

    cfg = CARTConfig()
    tuned = Path(a.config)
    if tuned.exists():
        best = json.loads(tuned.read_text())["pooled_best"]
        cfg = CARTConfig(tau=best["tau"], alpha=best["alpha"],
                         theta=best["theta"], lam=best["lam"])
    print("CART config:", cfg, flush=True)

    from cart.config import DEFAULT
    g2p = G2P()
    all_queries = read_jsonl(DATA / "queries.jsonl")
    report = {"config": cfg.__dict__, "model": a.model, "conditions": {}}

    for cond in a.conditions:
        corpus, asr = load_condition(a.model, cond)
        qs = filter_queries(all_queries, corpus, split=a.split)
        if not qs:
            print(f"[{cond}] no data, skipping", flush=True)
            continue
        systems = build_systems(corpus, asr, cfg, g2p,
                                include_dense=not a.no_dense,
                                dense_model=DEFAULT.dense_model)
        rows, rr = evaluate_systems(systems, qs)

        mask = ne_query_mask(qs, corpus)
        ne_rows = {n: {"mrr": round(float(v[mask].mean()), 4)} for n, v in rr.items()}
        gen_rows = {n: {"mrr": round(float(v[~mask].mean()), 4)} for n, v in rr.items()}

        report["conditions"][cond] = {
            "n_docs": len(corpus),
            "n_queries": len(qs),
            "asr": transcript_report(corpus, asr, tau=cfg.tau, theta=cfg.theta, g2p=g2p),
            "systems": rows,
            "ne_queries": {"n": int(mask.sum()), "mrr": ne_rows},
            "non_ne_queries": {"n": int((~mask).sum()), "mrr": gen_rows},
            "index_cost": index_cost(systems),
        }
        print(f"[{cond}] WER={report['conditions'][cond]['asr']['wer']:.3f} "
              f"bm25={rows['bm25_1best']['mrr']:.4f} "
              f"nbest={rows['nbest_uniform']['mrr']:.4f} "
              f"cart={rows['cart']['mrr']:.4f} "
              f"oracle={rows['oracle_bm25']['mrr']:.4f}", flush=True)
        np.save(RESULTS / f"rr_{cond}.npy", np.stack([rr[k] for k in sorted(rr)]))
        (RESULTS / f"rr_{cond}_systems.json").write_text(json.dumps(sorted(rr)))

    g2p.save()
    Path(a.out).write_text(json.dumps(report, indent=2))
    print("wrote", a.out)
