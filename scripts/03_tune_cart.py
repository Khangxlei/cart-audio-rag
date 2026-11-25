"""Tune CART's hyperparameters (tau, alpha, theta, lambda) on the dev split.

The search is factorised so we never rebuild an index we do not have to:
`s_text` depends on (tau, alpha), `s_phonetic` on (tau, theta), and lambda is a
pure post-hoc scalar on an already-computed matrix.  That turns a 5x3x3x7 grid
into 30 index builds instead of 315.
"""
import argparse, json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from cart.config import CARTConfig, DATA, RESULTS
from cart.data import read_jsonl
from cart.evaluate import filter_queries, load_condition
from cart.index import BM25Index, PhoneticIndex, content_terms
from cart.asr import tokenize
from cart.metrics import retrieval_metrics
from cart.phonetics import G2P
from cart.retrieval import build_documents

TAUS = [0.4, 0.5, 0.6, 0.7, 0.8]
ALPHAS = [0.25, 0.5, 1.0]
THETAS = [0.55, 0.65, 0.75]
LAMBDAS = [0.0, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0]


def rank_metrics(S, doc_ids, golds, topk=10):
    ranked = []
    for i in range(S.shape[0]):
        s = S[i]
        k = min(topk, len(s))
        idx = np.argpartition(-s, k - 1)[:k]
        idx = idx[np.argsort(-s[idx])]
        ranked.append([(doc_ids[j], float(s[j])) for j in idx])
    return retrieval_metrics(ranked, golds)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="small.en")
    ap.add_argument("--conditions", nargs="+", default=["clean", "snr5", "snr0", "snr-5"])
    ap.add_argument("--objective", default="mrr")
    a = ap.parse_args()

    g2p = G2P()
    all_queries = read_jsonl(DATA / "queries.jsonl")
    report = {}

    for cond in a.conditions:
        corpus, asr = load_condition(a.model, cond)
        qs = filter_queries(all_queries, corpus, split="dev")
        if not qs:
            continue
        golds = [q["gold_doc_id"] for q in qs]
        qterms = [content_terms(tokenize(q["question"])) for q in qs]
        doc_ids = [c["doc_id"] for c in corpus]
        rows = []

        for tau in TAUS:
            docs = build_documents(corpus, asr, "cart", tau, CARTConfig().max_alts)
            conf = np.array([d.conf for d in docs], dtype=np.float32)
            gate = 1.0 - conf

            # unexpanded index: supplies IDF and the exact-match test used by
            # the phonetic fallback, so phon_mats depend only on (tau, theta)
            idx_ref = BM25Index(docs, alpha=0.0)
            text_mats = {}
            for alpha in ALPHAS:
                bm = BM25Index(docs, alpha=alpha)
                text_mats[alpha] = np.stack([bm.score(q) for q in qterms])

            phon_mats = {}
            for theta in THETAS:
                pi = PhoneticIndex(docs, g2p, theta=theta)
                phon_mats[theta] = np.stack([
                    pi.score(q, idx_ref.term_idf, skip_mask_fn=idx_ref.contains)
                    for q in qterms])

            for alpha in ALPHAS:
                for theta in THETAS:
                    for lam in LAMBDAS:
                        S = text_mats[alpha] + lam * gate[None, :] * phon_mats[theta]
                        m = rank_metrics(S, doc_ids, golds)
                        rows.append({"tau": tau, "alpha": alpha, "theta": theta,
                                     "lam": lam, **m})
            print(f"[{cond}] tau={tau} done", flush=True)

        rows.sort(key=lambda r: -r[a.objective])
        report[cond] = {"best": rows[0], "top10": rows[:10], "grid": rows}
        print(f"[{cond}] best: {rows[0]}", flush=True)

    g2p.save()
    # one configuration for all conditions: best mean rank of the objective
    keys, agg = None, {}
    for cond, r in report.items():
        for row in r["grid"]:
            k = (row["tau"], row["alpha"], row["theta"], row["lam"])
            agg.setdefault(k, []).append(row[a.objective])
    pooled = sorted(agg.items(), key=lambda kv: -np.mean(kv[1]))
    best = pooled[0][0]
    report["pooled_best"] = {"tau": best[0], "alpha": best[1], "theta": best[2],
                             "lam": best[3], f"mean_{a.objective}": round(float(np.mean(pooled[0][1])), 4)}
    (RESULTS / "tuning.json").write_text(json.dumps(report, indent=2))
    print("POOLED BEST:", json.dumps(report["pooled_best"], indent=2))
