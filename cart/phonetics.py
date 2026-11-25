"""Grapheme-to-phoneme conversion and articulatory-feature phonetic distance.

`g2p_en` gives ARPAbet.  A plain edit distance over phone symbols treats
"S EH L" vs "S EY L" the same as "S EH L" vs "S P L", which is wrong: EH and EY
are both front vowels and confusable, P is not.  We therefore score
substitutions by articulatory-feature overlap, so acoustically confusable pairs
(cell/sail, Lorenz/Lawrence) come out close and unrelated pairs do not.
"""
from __future__ import annotations

import json
import pickle
import re
from functools import lru_cache
from pathlib import Path

from .config import DATA

_STRESS = re.compile(r"\d")
_LONG_DIGITS = re.compile(r"\d{10,}")
_MAX_G2P_LEN = 64


def _is_pathological(tok: str) -> bool:
    """Tokens no real pronunciation exists for: ASR degeneration artifacts."""
    return (not tok) or len(tok) > 25 or bool(_LONG_DIGITS.search(tok))

# (place, manner, voiced) -- place/manner on coarse ordinal scales
_CONS = {
    "P": (1, 1, 0), "B": (1, 1, 1), "T": (3, 1, 0), "D": (3, 1, 1),
    "K": (6, 1, 0), "G": (6, 1, 1),
    "F": (2, 2, 0), "V": (2, 2, 1), "TH": (3, 2, 0), "DH": (3, 2, 1),
    "S": (3, 2, 0), "Z": (3, 2, 1), "SH": (4, 2, 0), "ZH": (4, 2, 1),
    "HH": (7, 2, 0),
    "CH": (4, 3, 0), "JH": (4, 3, 1),
    "M": (1, 4, 1), "N": (3, 4, 1), "NG": (6, 4, 1),
    "L": (3, 5, 1), "R": (4, 5, 1), "W": (1, 6, 1), "Y": (5, 6, 1),
}
# (height, backness, rounded, diphthong)
_VOW = {
    "IY": (3, 0, 0, 0), "IH": (2, 0, 0, 0), "EY": (2, 0, 0, 1), "EH": (1, 0, 0, 0),
    "AE": (0, 0, 0, 0), "AA": (0, 2, 0, 0), "AO": (1, 2, 1, 0), "OW": (2, 2, 1, 1),
    "UH": (2, 2, 1, 0), "UW": (3, 2, 1, 0), "AH": (1, 1, 0, 0), "ER": (1, 1, 0, 0),
    "AY": (0, 1, 0, 1), "AW": (0, 1, 1, 1), "OY": (1, 2, 1, 1),
}


@lru_cache(maxsize=None)
def phone_sub_cost(a: str, b: str) -> float:
    if a == b:
        return 0.0
    if a in _VOW and b in _VOW:
        fa, fb = _VOW[a], _VOW[b]
        d = abs(fa[0] - fb[0]) / 3 + abs(fa[1] - fb[1]) / 2 + abs(fa[2] - fb[2]) + abs(fa[3] - fb[3])
        return 0.25 + 0.55 * min(d / 4.0, 1.0)
    if a in _CONS and b in _CONS:
        fa, fb = _CONS[a], _CONS[b]
        d = abs(fa[0] - fb[0]) / 6 + (0.0 if fa[1] == fb[1] else 1.0) + abs(fa[2] - fb[2])
        return 0.25 + 0.65 * min(d / 3.0, 1.0)
    return 1.0          # vowel <-> consonant: maximally different


def _indel_cost(p: str) -> float:
    # dropping/adding an unstressed schwa or a glide is cheap; it happens a lot
    return 0.55 if p in ("AH", "ER", "Y", "W", "HH") else 0.9


class G2P:
    """Cached grapheme-to-phoneme. CMUdict lookups are exact; OOV words fall
    back to g2p_en's neural model, which is slow, hence the persistent cache."""

    def __init__(self, cache_path: Path = DATA / "g2p_cache.pkl"):
        self.cache_path = cache_path
        self.cache: dict[str, tuple] = {}
        if cache_path.exists():
            try:
                self.cache = pickle.loads(cache_path.read_bytes())
            except Exception:
                self.cache = {}
        self._g2p = None

    def __call__(self, word: str) -> tuple:
        w = word.lower()
        hit = self.cache.get(w)
        if hit is not None:
            return hit
        self._ensure_g2p()
        # g2p_en spells numbers out via `inflect`, which raises
        # NumOutOfRangeError past ~10^303. Whisper degenerates into long runs of
        # digits on noisy audio (we see tokens of 300+ zeros), so clamp before
        # conversion and tolerate a failure on such a token rather than letting
        # one bad token kill the whole run.
        safe = _LONG_DIGITS.sub(lambda m: m.group(0)[:9], w)[:_MAX_G2P_LEN]
        try:
            phones = tuple(_STRESS.sub("", p) for p in self._g2p(safe)
                           if p.strip() and p.isalnum())
        except Exception:
            # Only pathological tokens get a free pass. A failure on an ordinary
            # word means the environment is broken (e.g. missing NLTK data), and
            # swallowing it would silently disable the whole phonetic channel
            # while every downstream number still looked plausible.
            if _is_pathological(w):
                phones = ()
            else:
                raise
        self.cache[w] = phones
        return phones

    def _ensure_g2p(self):
        """Load g2p_en and prove it actually works before anything depends on it."""
        if self._g2p is not None:
            return
        from g2p_en import G2p
        self._g2p = G2p()
        probe = tuple(p for p in self._g2p("cell") if p.strip() and p.isalnum())
        if not probe:
            self._g2p = None
            raise RuntimeError(
                "g2p_en produced no phonemes for the word 'cell'. The NLTK data "
                "it needs is probably missing (averaged_perceptron_tagger_eng, "
                "cmudict). Fix NLTK_DATA rather than continuing: CART's phonetic "
                "channel would silently score zero for every term."
            )

    def save(self):
        self.cache_path.write_bytes(pickle.dumps(self.cache))


def weighted_edit(a: tuple, b: tuple) -> float:
    if a == b:
        return 0.0
    if not a or not b:
        return float(sum(_indel_cost(p) for p in (a or b)))
    prev = [0.0] * (len(b) + 1)
    for j, pb in enumerate(b, 1):
        prev[j] = prev[j - 1] + _indel_cost(pb)
    for i, pa in enumerate(a, 1):
        cur = [prev[0] + _indel_cost(pa)] + [0.0] * len(b)
        for j, pb in enumerate(b, 1):
            cur[j] = min(prev[j] + _indel_cost(pa),
                         cur[j - 1] + _indel_cost(pb),
                         prev[j - 1] + phone_sub_cost(pa, pb))
        prev = cur
    return prev[-1]


def phonetic_similarity(a: tuple, b: tuple) -> float:
    """1 - normalised weighted edit distance, clipped to [0, 1]."""
    if not a and not b:
        return 1.0
    denom = max(sum(_indel_cost(p) for p in a), sum(_indel_cost(p) for p in b), 1e-9)
    return max(0.0, 1.0 - weighted_edit(a, b) / denom)
