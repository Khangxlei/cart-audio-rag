"""Retrieval and question-answering metrics."""
from __future__ import annotations

import re
import string
from collections import Counter

import numpy as np


def retrieval_metrics(ranked_lists, gold_doc_ids, ks=(1, 5, 10)) -> dict:
    """Spoken-SQuAD has exactly one relevant passage per question, so
    Recall@k == Hit@k and nDCG@10 reduces to 1/log2(rank+1)."""
    out = {f"recall@{k}": 0.0 for k in ks}
    mrr = ndcg = 0.0
    n = len(gold_doc_ids)
    for ranked, gold in zip(ranked_lists, gold_doc_ids):
        ids = [d for d, _ in ranked]
        rank = ids.index(gold) + 1 if gold in ids else None
        for k in ks:
            if rank is not None and rank <= k:
                out[f"recall@{k}"] += 1
        if rank is not None:
            mrr += 1.0 / rank
            if rank <= 10:
                ndcg += 1.0 / np.log2(rank + 1)
    for k in ks:
        out[f"recall@{k}"] /= max(n, 1)
    out["mrr"] = mrr / max(n, 1)
    out["ndcg@10"] = ndcg / max(n, 1)
    return {k: round(float(v), 4) for k, v in out.items()}


# ---- SQuAD official answer normalisation ------------------------------------
_ARTICLES = re.compile(r"\b(a|an|the)\b")


def _norm(s: str) -> str:
    s = s.lower()
    s = "".join(ch for ch in s if ch not in set(string.punctuation))
    s = _ARTICLES.sub(" ", s)
    return " ".join(s.split())


def exact_match(pred: str, golds) -> float:
    return float(any(_norm(pred) == _norm(g) for g in golds))


def token_f1(pred: str, golds) -> float:
    best = 0.0
    pt = _norm(pred).split()
    if not pt:
        return 0.0
    for g in golds:
        gt = _norm(g).split()
        if not gt:
            continue
        common = Counter(pt) & Counter(gt)
        ns = sum(common.values())
        if ns == 0:
            continue
        p, r = ns / len(pt), ns / len(gt)
        best = max(best, 2 * p * r / (p + r))
    return best


def qa_metrics(preds, gold_answers) -> dict:
    em = float(np.mean([exact_match(p, g) for p, g in zip(preds, gold_answers)]))
    f1 = float(np.mean([token_f1(p, g) for p, g in zip(preds, gold_answers)]))
    return {"em": round(em, 4), "f1": round(f1, 4)}


def bootstrap_ci(values, n_boot=2000, seed=0, alpha=0.05):
    """Percentile bootstrap CI over per-query values."""
    v = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(v), size=(n_boot, len(v)))
    means = v[idx].mean(axis=1)
    lo, hi = np.quantile(means, [alpha / 2, 1 - alpha / 2])
    return round(float(v.mean()), 4), round(float(lo), 4), round(float(hi), 4)


def paired_bootstrap_p(a, b, n_boot=5000, seed=0):
    """Two-sided paired bootstrap p-value for mean(a) - mean(b)."""
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    d = a - b
    obs = d.mean()
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(d), size=(n_boot, len(d)))
    boot = d[idx].mean(axis=1) - obs
    return round(float((np.abs(boot) >= abs(obs)).mean()), 4)
