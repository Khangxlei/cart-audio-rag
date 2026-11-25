"""Shared evaluation helpers: system construction, running queries, scoring."""
from __future__ import annotations

import json
import time
from dataclasses import replace

import numpy as np

from .asr import load_asr
from .config import CARTConfig, DATA, RUNS
from .data import read_jsonl
from .metrics import bootstrap_ci, paired_bootstrap_p, retrieval_metrics
from .phonetics import G2P
from .retrieval import DenseRetriever, make_retriever


def load_condition(model: str, condition: str, corpus=None):
    """Return (corpus restricted to transcribed docs, asr_by_doc)."""
    corpus = corpus if corpus is not None else read_jsonl(DATA / "corpus.jsonl")
    asr = load_asr(RUNS / "asr" / model / f"{condition}.jsonl")
    corpus = [c for c in corpus if c["doc_id"] in asr]
    return corpus, asr


def filter_queries(queries, corpus, split=None):
    ok = {c["doc_id"] for c in corpus}
    return [q for q in queries
            if q["gold_doc_id"] in ok and (split is None or q["split"] == split)]


def run_queries(ret, queries, topk=10):
    t0 = time.time()
    ranked = [ret.search(q["question"], topk) for q in queries]
    return ranked, (time.time() - t0) / max(len(queries), 1) * 1000  # ms/query


def per_query_rr(ranked_lists, queries):
    """Reciprocal rank per query, for CIs and paired significance tests."""
    out = []
    for ranked, q in zip(ranked_lists, queries):
        ids = [d for d, _ in ranked]
        out.append(1.0 / (ids.index(q["gold_doc_id"]) + 1) if q["gold_doc_id"] in ids else 0.0)
    return np.array(out)


def per_query_hit(ranked_lists, queries, k=1):
    out = []
    for ranked, q in zip(ranked_lists, queries):
        ids = [d for d, _ in ranked][:k]
        out.append(1.0 if q["gold_doc_id"] in ids else 0.0)
    return np.array(out)


def build_systems(corpus, asr, cfg: CARTConfig, g2p: G2P, cfgx=None,
                  include_dense=True, dense_model=None):
    """The baseline suite plus CART, as named (system -> retriever) pairs."""
    systems = {
        "oracle_bm25": make_retriever("oracle_bm25", corpus, asr, "oracle", cfg, g2p),
        "bm25_1best": make_retriever("bm25_1best", corpus, asr, "1best", cfg, g2p),
        "nbest_uniform": make_retriever("nbest_uniform", corpus, asr, "nbest_uniform", cfg, g2p),
        "cart": make_retriever("cart", corpus, asr, "cart", cfg, g2p),
    }
    if include_dense and dense_model:
        systems["dense_1best"] = DenseRetriever(
            "dense_1best", corpus, asr, "1best", dense_model)
    return systems


def evaluate_systems(systems, queries, topk=10, ref="bm25_1best"):
    rows, rr = {}, {}
    for name, ret in systems.items():
        ranked, ms = run_queries(ret, queries, topk)
        m = retrieval_metrics(ranked, [q["gold_doc_id"] for q in queries])
        rr[name] = per_query_rr(ranked, queries)
        mean, lo, hi = bootstrap_ci(rr[name])
        m["mrr_ci"] = [lo, hi]
        m["ms_per_query"] = round(ms, 2)
        rows[name] = m
    for name in rows:
        if name != ref and ref in rr:
            rows[name]["p_vs_" + ref] = paired_bootstrap_p(rr[name], rr[ref])
    return rows, rr
