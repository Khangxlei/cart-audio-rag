"""Index structures for CART.

Three views over the same collection:

* text index      -- BM25 over the 1-best transcript, optionally *expanded*
                     with beam alternatives for low-confidence words.  Expanded
                     postings carry weight `alpha` and do NOT inflate the
                     document length, so expansion cannot be gamed by simply
                     making documents longer.
* phonetic index  -- phoneme strings of the low-confidence terms only, so a
                     query term that no document spells correctly can still
                     match the span the recognizer was unsure about.
* confidence      -- per-document mean word probability `c`, used to gate how
                     much the phonetic channel is trusted.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field

import numpy as np

from .phonetics import G2P, phonetic_similarity

STOPWORDS = set("""a an and are as at be been by for from had has have he her his i in is it its
of on or that the this to was were what when where which who whom why will with how did does do
there their they you your not no s t re ve ll d m o""".split())


# Whisper degenerates into very long digit runs on noisy audio (tokens of 300+
# characters). They are never useful index or query terms, and they are
# pathological for both BM25 statistics and grapheme-to-phoneme conversion.
MAX_TERM_LEN = 25


def indexable(t: str) -> bool:
    return 1 < len(t) <= MAX_TERM_LEN


def content_terms(tokens):
    return [t for t in tokens if t not in STOPWORDS and indexable(t)]


@dataclass
class Document:
    doc_id: str
    tokens: list[str]                       # 1-best (or gold) tokens
    alt_tokens: list[str] = field(default_factory=list)   # expansion postings
    lowconf_terms: set = field(default_factory=set)       # phonetic-index terms
    conf: float = 1.0                       # chunk confidence c


class BM25Index:
    def __init__(self, docs: list[Document], k1=0.9, b=0.4, alpha=0.5):
        self.docs = docs
        self.k1, self.b, self.alpha = k1, b, alpha
        self.doc_ids = [d.doc_id for d in docs]
        self.N = len(docs)
        self.dl = np.array([max(len(d.tokens), 1) for d in docs], dtype=np.float32)
        self.avgdl = float(self.dl.mean())

        postings: dict[str, dict[int, float]] = defaultdict(dict)
        df: Counter = Counter()
        for i, d in enumerate(docs):
            prim = Counter(d.tokens)
            alts = Counter(d.alt_tokens)
            for t, c in prim.items():
                postings[t][i] = float(c)
            for t, c in alts.items():
                postings[t][i] = postings[t].get(i, 0.0) + alpha * c
            for t in set(prim) | set(alts):
                df[t] += 1
        self.postings = {t: (np.fromiter(p.keys(), dtype=np.int32),
                             np.fromiter(p.values(), dtype=np.float32))
                         for t, p in postings.items()}
        self.idf = {t: float(np.log(1 + (self.N - n + 0.5) / (n + 0.5))) for t, n in df.items()}

    def term_idf(self, t: str) -> float:
        n = 0
        return self.idf.get(t, float(np.log(1 + (self.N + 0.5) / 0.5)))

    def score(self, query_terms) -> np.ndarray:
        s = np.zeros(self.N, dtype=np.float32)
        denom_len = self.k1 * (1 - self.b + self.b * self.dl / self.avgdl)
        for t in query_terms:
            p = self.postings.get(t)
            if p is None:
                continue
            idx, tf = p
            s[idx] += self.idf[t] * (tf * (self.k1 + 1)) / (tf + denom_len[idx])
        return s

    def contains(self, t: str) -> np.ndarray:
        """Boolean mask of documents whose *text* index holds term t."""
        m = np.zeros(self.N, dtype=bool)
        p = self.postings.get(t)
        if p is not None:
            m[p[0]] = True
        return m


class PhoneticIndex:
    """Phoneme index over low-confidence terms, with phoneme-bigram blocking so
    a query term is only compared against plausibly similar vocabulary."""

    def __init__(self, docs: list[Document], g2p: G2P, theta: float = 0.6):
        self.g2p, self.theta, self.N = g2p, theta, len(docs)
        self.term_docs: dict[str, list[int]] = defaultdict(list)
        for i, d in enumerate(docs):
            for t in d.lowconf_terms:
                if indexable(t):
                    self.term_docs[t].append(i)
        self.phones = {t: g2p(t) for t in self.term_docs}
        self.bigrams: dict[tuple, set] = defaultdict(set)
        for t, ph in self.phones.items():
            for bg in self._bigrams(ph):
                self.bigrams[bg].add(t)
        self._match_cache: dict[str, dict[str, float]] = {}

    @staticmethod
    def _bigrams(ph):
        if len(ph) < 2:
            return [(ph[0], "#")] if ph else []
        return list(zip(ph, ph[1:]))

    def match(self, term: str) -> dict[str, float]:
        """Low-confidence vocabulary phonetically similar to `term`."""
        if term in self._match_cache:
            return self._match_cache[term]
        ph = self.g2p(term)
        cands = set()
        for bg in self._bigrams(ph):
            cands |= self.bigrams.get(bg, set())
        out = {}
        for t in cands:
            if t == term:
                continue
            tp = self.phones[t]
            if abs(len(tp) - len(ph)) > 3:
                continue
            s = phonetic_similarity(ph, tp)
            if s >= self.theta:
                out[t] = s
        self._match_cache[term] = out
        return out

    def score(self, query_terms, idf_fn, skip_mask_fn=None) -> np.ndarray:
        """IDF-weighted, per-query-term max phonetic match, on a BM25-like scale.

        `skip_mask_fn(term) -> bool mask` marks documents that already match the
        term exactly.  The phonetic channel is a *fallback*, so it is suppressed
        there and only fires where surface matching failed.
        """
        s = np.zeros(self.N, dtype=np.float32)
        for q in query_terms:
            m = self.match(q)
            if not m:
                continue
            w = idf_fn(q)
            best = np.zeros(self.N, dtype=np.float32)
            for t, sim in m.items():
                # rescale theta..1 -> 0..1 so a barely-passing match counts little
                v = (sim - self.theta) / max(1e-6, 1 - self.theta)
                idx = self.term_docs[t]
                np.maximum.at(best, idx, v)
            if skip_mask_fn is not None:
                best[skip_mask_fn(q)] = 0.0
            s += w * best
        return s
