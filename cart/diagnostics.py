"""Diagnostics that explain *why* CART does or does not help.

These answer the three empirical questions the proposal's "Potential
Challenges" section raises:

1. Does Whisper's word confidence actually find its own errors? (calibration)
2. Do the beam alternatives ever contain the correct word? (alt coverage)
3. Is the correct word phonetically reachable from the misrecognition?
"""
from __future__ import annotations

import difflib
import re
from collections import Counter

import numpy as np

from .asr import tokenize
from .phonetics import phonetic_similarity

_NUM = re.compile(r"^\d[\d,.]*$")


def named_entity_tokens(gold_text: str) -> set[str]:
    """Heuristic named-entity / rare-term set: capitalised words that are not
    sentence-initial, plus numerals.  Used only for *analysis* -- CART itself
    never runs an NER, as the proposal requires."""
    out = set()
    sent_start = True
    for raw in gold_text.split():
        tok = tokenize(raw)
        tok = tok[0] if tok else ""
        if tok and not sent_start and (raw[0].isupper() or _NUM.match(raw)):
            out.add(tok)
        sent_start = raw.endswith((".", "!", "?"))
    return out


def align(gold: list[str], hyp: list[str]):
    """Yield (op, gold_token_or_None, hyp_index_or_None)."""
    sm = difflib.SequenceMatcher(a=gold, b=hyp, autojunk=False)
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            for k in range(i2 - i1):
                yield "ok", gold[i1 + k], j1 + k
        elif tag == "replace":
            n = max(i2 - i1, j2 - j1)
            for k in range(n):
                g = gold[i1 + k] if i1 + k < i2 else None
                h = j1 + k if j1 + k < j2 else None
                yield ("sub" if g is not None and h is not None else
                       ("del" if h is None else "ins")), g, h
        elif tag == "delete":
            for k in range(i1, i2):
                yield "del", gold[k], None
        else:
            for k in range(j1, j2):
                yield "ins", None, k


def transcript_report(corpus, asr_by_doc, tau=0.6, theta=0.6, g2p=None) -> dict:
    """Per-condition WER / NE-WER / confidence calibration / alternative coverage."""
    n_err = n_ref = ne_err = ne_ref = 0
    conf_ok, conf_bad = [], []
    low_total = low_with_alt = 0
    sub_total = sub_alt_hit = sub_phon_reach = 0
    sub_lowconf = 0
    alt_postings = 0

    for c in corpus:
        r = asr_by_doc.get(c["doc_id"])
        if r is None:
            continue
        gold = tokenize(c["gold_text"])
        hyp = [w.word for w in r.words]
        nes = named_entity_tokens(c["gold_text"])
        for w in r.words:
            if w.prob < tau:
                low_total += 1
                if w.alts:
                    low_with_alt += 1
                alt_postings += len(w.alts[:3])

        for op, g, hi in align(gold, hyp):
            if g is not None:
                n_ref += 1
                is_ne = g in nes
                ne_ref += is_ne
                if op != "ok":
                    n_err += 1
                    ne_err += is_ne
            if hi is None:
                continue
            w = r.words[hi]
            (conf_ok if op == "ok" else conf_bad).append(w.prob)
            if op == "sub" and g is not None:
                sub_total += 1
                if w.prob < tau:
                    sub_lowconf += 1
                if g in w.alts:
                    sub_alt_hit += 1
                if g2p is not None and phonetic_similarity(g2p(g), g2p(w.word)) >= theta:
                    sub_phon_reach += 1

    def _auc(pos, neg):
        """P(confidence of a correct word > confidence of an error)."""
        if not pos or not neg:
            return float("nan")
        a = np.asarray(pos)
        b = np.asarray(neg)
        order = np.argsort(np.concatenate([a, b]), kind="mergesort")
        ranks = np.empty(len(order), dtype=np.float64)
        ranks[order] = np.arange(1, len(order) + 1)
        return float((ranks[: len(a)].sum() - len(a) * (len(a) + 1) / 2) / (len(a) * len(b)))

    return {
        "wer": round(n_err / max(n_ref, 1), 4),
        "ne_wer": round(ne_err / max(ne_ref, 1), 4),
        "non_ne_wer": round((n_err - ne_err) / max(n_ref - ne_ref, 1), 4),
        "ne_token_share": round(ne_ref / max(n_ref, 1), 4),
        "mean_conf_correct": round(float(np.mean(conf_ok)) if conf_ok else 0.0, 4),
        "mean_conf_error": round(float(np.mean(conf_bad)) if conf_bad else 0.0, 4),
        "conf_auc": round(_auc(conf_ok, conf_bad), 4),
        "pct_words_lowconf": round(low_total / max(len(conf_ok) + len(conf_bad), 1), 4),
        "alt_coverage_lowconf": round(low_with_alt / max(low_total, 1), 4),
        "alt_postings_per_doc": round(alt_postings / max(len(corpus), 1), 2),
        # of all substitution errors ...
        "sub_errors": sub_total,
        "sub_flagged_lowconf": round(sub_lowconf / max(sub_total, 1), 4),
        "sub_recovered_by_alt": round(sub_alt_hit / max(sub_total, 1), 4),
        "sub_phonetically_reachable": round(sub_phon_reach / max(sub_total, 1), 4),
    }


def index_cost(retrievers: dict) -> dict:
    out = {}
    for name, r in retrievers.items():
        bm25 = getattr(r, "bm25", None)
        if bm25 is None:
            continue
        n_post = sum(len(v[0]) for v in bm25.postings.values())
        out[name] = {
            "vocab": len(bm25.postings),
            "postings": n_post,
            "phonetic_terms": len(r.phon.term_docs) if getattr(r, "phon", None) else 0,
        }
    return out
