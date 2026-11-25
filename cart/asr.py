"""Faster-Whisper wrapper that exposes word confidences *and* N-best beams.

faster-whisper only hands back the single best beam.  The beams are right there
in CTranslate2's result object though, so we wrap `WhisperModel.model.generate`,
ask CTranslate2 for `num_hypotheses == beam_size`, and record every hypothesis.
Hypothesis 0 is byte-identical to what faster-whisper returns, so concatenating
hypothesis k over the 30s windows yields a complete alternative transcript that
is word-alignable against the 1-best.  Aligning them (difflib) recovers, for
each 1-best word position, the spellings the other beams proposed there --
exactly the "I bought a new cell / sail / shell" structure CART indexes.
"""
from __future__ import annotations

import difflib
import re
from dataclasses import dataclass, field

import numpy as np

from .config import ASRConfig

_PUNCT = re.compile(r"[^\w']+")


def normalize(tok: str) -> str:
    return _PUNCT.sub("", tok.lower()).strip("'")


def tokenize(text: str) -> list[str]:
    return [t for t in (normalize(w) for w in text.split()) if t]


@dataclass
class ASRWord:
    word: str
    start: float
    end: float
    prob: float
    alts: list[str] = field(default_factory=list)


@dataclass
class ASRResult:
    doc_id: str
    condition: str
    text: str
    words: list[ASRWord]
    nbest: list[str]            # full alternative transcripts, hypothesis 1..K-1
    mean_conf: float

    def to_json(self):
        return {
            "doc_id": self.doc_id,
            "condition": self.condition,
            "text": self.text,
            "mean_conf": self.mean_conf,
            "nbest": self.nbest,
            "words": [
                {"w": w.word, "s": round(w.start, 2), "e": round(w.end, 2),
                 "p": round(w.prob, 4), "alts": w.alts}
                for w in self.words
            ],
        }

    @staticmethod
    def from_json(d):
        return ASRResult(
            d["doc_id"], d["condition"], d["text"],
            [ASRWord(w["w"], w["s"], w["e"], w["p"], w.get("alts", [])) for w in d["words"]],
            d.get("nbest", []), d["mean_conf"],
        )


class NBestWhisper:
    def __init__(self, cfg: ASRConfig):
        from faster_whisper import WhisperModel

        self.cfg = cfg
        self.model = WhisperModel(cfg.model, device=cfg.device, compute_type=cfg.compute_type)
        self._tokenizer = None

    # ------------------------------------------------------------------
    def _decode(self, tokens) -> str:
        return self._tokenizer.decode([t for t in tokens if t < self._tokenizer.eot])

    def transcribe(self, audio: np.ndarray, doc_id: str, condition: str) -> ASRResult:
        cfg = self.cfg
        captured: list = []
        inner = self.model.model

        # The CTranslate2 Whisper object is a C extension whose attributes are
        # read-only, so we swap in a delegating proxy instead of patching it.
        class _Proxy:
            def __getattr__(self, name):
                return getattr(inner, name)

            def generate(self, encoder_output, prompts, **kw):
                if kw.get("beam_size", 1) > 1:
                    kw["num_hypotheses"] = cfg.num_hypotheses
                res = inner.generate(encoder_output, prompts, **kw)
                captured.append(res[0])
                return res

        self.model.model = _Proxy()
        try:
            segments, _info = self.model.transcribe(
                audio,
                language=cfg.language,
                beam_size=cfg.beam_size,
                best_of=cfg.beam_size,
                temperature=cfg.temperature,
                word_timestamps=cfg.word_timestamps,
                condition_on_previous_text=cfg.condition_on_previous_text,
                vad_filter=False,
            )
            segments = list(segments)
        finally:
            self.model.model = inner

        if self._tokenizer is None:
            from faster_whisper.tokenizer import Tokenizer
            self._tokenizer = Tokenizer(
                self.model.hf_tokenizer, inner.is_multilingual,
                task="transcribe", language=cfg.language,
            )

        words: list[ASRWord] = []
        for seg in segments:
            for w in (seg.words or []):
                tok = normalize(w.word)
                if tok:
                    words.append(ASRWord(tok, w.start, w.end, float(w.probability)))
        text = " ".join(s.text.strip() for s in segments).strip()

        # ---- reconstruct the K-1 alternative transcripts ------------------
        nbest_texts: list[str] = []
        for k in range(1, cfg.num_hypotheses):
            parts = []
            for res in captured:
                if k < len(res.sequences_ids):
                    parts.append(self._decode(res.sequences_ids[k]).strip())
            if parts:
                nbest_texts.append(" ".join(parts).strip())

        self._attach_alternatives(words, nbest_texts)
        mean_conf = float(np.mean([w.prob for w in words])) if words else 0.0
        return ASRResult(doc_id, condition, text, words, nbest_texts, mean_conf)

    # ------------------------------------------------------------------
    @staticmethod
    def _attach_alternatives(words: list[ASRWord], nbest_texts: list[str]) -> None:
        """Word-align each alternative transcript to the 1-best and record, for
        every 1-best position, the spellings a competing beam proposed there."""
        best = [w.word for w in words]
        if not best:
            return
        for alt_text in nbest_texts:
            alt = tokenize(alt_text)
            if not alt:
                continue
            sm = difflib.SequenceMatcher(a=best, b=alt, autojunk=False)
            for tag, i1, i2, j1, j2 in sm.get_opcodes():
                if tag == "equal":
                    continue
                span = alt[j1:j2]
                if not span or (i2 - i1) == 0:
                    continue
                # A 1:1 substitution is the clean case ("cell" -> "sail").
                # For ragged spans we attach the whole competing span to every
                # 1-best word inside it; the index treats them as a bag anyway.
                if (i2 - i1) == len(span):
                    pairs = zip(range(i1, i2), span)
                else:
                    pairs = ((i, s) for i in range(i1, i2) for s in span)
                for i, s in pairs:
                    if s and s != words[i].word and s not in words[i].alts:
                        words[i].alts.append(s)


def load_asr(path) -> dict:
    """doc_id -> ASRResult from a runs/asr/<model>/<condition>.jsonl file."""
    import json
    out = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                r = ASRResult.from_json(json.loads(line))
                out[r.doc_id] = r
    return out
