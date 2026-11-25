"""Retrievers: the CART scorer and the baselines it is compared against."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from .asr import ASRResult, tokenize
from .config import CARTConfig
from .index import BM25Index, Document, PhoneticIndex, content_terms
from .phonetics import G2P

MODES = ("oracle", "1best", "nbest_uniform", "cart")


def build_documents(corpus, asr_by_doc, mode: str, tau: float, max_alts: int):
    """Turn ASR output into the document view that a given system indexes."""
    docs = []
    for c in corpus:
        did = c["doc_id"]
        if mode == "oracle":
            docs.append(Document(did, tokenize(c["gold_text"])))
            continue
        r = asr_by_doc[did]
        toks = [w.word for w in r.words]
        if mode == "1best":
            docs.append(Document(did, toks, conf=r.mean_conf))
        elif mode == "nbest_uniform":
            # index every beam hypothesis wholesale, with no confidence targeting
            base = set(toks)
            alts = [t for h in r.nbest for t in tokenize(h) if t not in base]
            docs.append(Document(did, toks, alt_tokens=alts, conf=r.mean_conf))
        elif mode == "cart":
            low = [w for w in r.words if w.prob < tau]
            alts, lowconf = [], set()
            for w in low:
                lowconf.add(w.word)
                for a in w.alts[:max_alts]:
                    alts.append(a)
                    lowconf.add(a)
            docs.append(Document(did, toks, alt_tokens=alts,
                                 lowconf_terms=lowconf, conf=r.mean_conf))
        else:
            raise ValueError(mode)
    return docs


@dataclass
class Retriever:
    name: str
    bm25: BM25Index
    phon: Optional[PhoneticIndex] = None
    conf: Optional[np.ndarray] = None
    cfg: Optional[CARTConfig] = None
    # Unexpanded (1-best only) index. IDF and the "already matched" test for the
    # phonetic fallback are taken from it, so the fallback fires exactly when the
    # *transcript* misses the query term, independent of how the expansion is
    # weighted.
    ref: Optional[BM25Index] = None

    def scores(self, question: str) -> np.ndarray:
        q = content_terms(tokenize(question))
        s = self.bm25.score(q)
        if self.phon is not None and self.cfg is not None and self.cfg.use_phonetic:
            ref = self.ref or self.bm25
            sp = self.phon.score(q, ref.term_idf, skip_mask_fn=ref.contains)
            gate = (1.0 - self.conf) if self.cfg.confidence_gate else 1.0
            s = s + self.cfg.lam * gate * sp
        return s

    def search(self, question: str, topk: int = 10):
        s = self.scores(question)
        k = min(topk, len(s))
        idx = np.argpartition(-s, k - 1)[:k]
        idx = idx[np.argsort(-s[idx])]
        return [(self.bm25.doc_ids[i], float(s[i])) for i in idx]


def make_retriever(name, corpus, asr_by_doc, mode, cfg: CARTConfig, g2p=None):
    docs = build_documents(corpus, asr_by_doc, mode, cfg.tau, cfg.max_alts)
    use_alt = cfg.use_alt_index or mode == "nbest_uniform"
    alpha = cfg.alpha if (mode in ("cart", "nbest_uniform") and use_alt) else 0.0
    bm25 = BM25Index(docs, k1=cfg.bm25_k1, b=cfg.bm25_b, alpha=alpha)
    phon = ref = None
    if mode == "cart" and cfg.use_phonetic:
        phon = PhoneticIndex(docs, g2p or G2P(), theta=cfg.theta)
        ref = bm25 if alpha == 0.0 else BM25Index(docs, k1=cfg.bm25_k1, b=cfg.bm25_b, alpha=0.0)
    conf = np.array([d.conf for d in docs], dtype=np.float32)
    return Retriever(name, bm25, phon, conf, cfg if mode == "cart" else None, ref)


class DenseRetriever:
    """Neural dense retrieval over the 1-best transcript.

    Stands in for the proposal's SpeechRAG baseline: SpeechRAG requires a
    trained speech encoder that has not been publicly released, so it cannot be
    reproduced here.  This keeps a *learned* retriever in the comparison, which
    is the property that made SpeechRAG informative as a baseline.
    """

    def __init__(self, name, corpus, asr_by_doc, mode, model_name, device="cuda"):
        from sentence_transformers import SentenceTransformer

        self.name = name
        self.doc_ids = [c["doc_id"] for c in corpus]
        if mode == "oracle":
            texts = [c["gold_text"] for c in corpus]
        else:
            texts = [asr_by_doc[c["doc_id"]].text for c in corpus]
        self.m = SentenceTransformer(model_name, device=device)
        self.emb = self.m.encode(texts, batch_size=32, convert_to_numpy=True,
                                 normalize_embeddings=True, show_progress_bar=False)

    def scores(self, question: str) -> np.ndarray:
        q = self.m.encode([question], convert_to_numpy=True, normalize_embeddings=True)
        return (self.emb @ q[0]).astype(np.float32)

    def search(self, question: str, topk: int = 10):
        s = self.scores(question)
        k = min(topk, len(s))
        idx = np.argpartition(-s, k - 1)[:k]
        idx = idx[np.argsort(-s[idx])]
        return [(self.doc_ids[i], float(s[i])) for i in idx]
