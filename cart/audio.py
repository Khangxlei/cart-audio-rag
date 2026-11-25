"""Audio loading and controlled degradation (noise-robustness axis)."""
from __future__ import annotations

import numpy as np
import soundfile as sf

SR = 16000


def load(path) -> np.ndarray:
    x, sr = sf.read(path, dtype="float32", always_2d=False)
    if x.ndim > 1:
        x = x.mean(axis=1)
    assert sr == SR, f"expected {SR} Hz, got {sr}"
    return x


def _babble(n: int, rng: np.random.Generator) -> np.ndarray:
    """Cheap speech-shaped (babble-like) noise: white noise with a 1/f-ish tilt
    and a formant-band emphasis, which stresses an ASR far more than white noise."""
    w = rng.standard_normal(n).astype(np.float32)
    spec = np.fft.rfft(w)
    f = np.fft.rfftfreq(n, 1 / SR)
    tilt = 1.0 / np.sqrt(np.maximum(f, 50.0) / 50.0)          # pink tilt
    emph = 1.0 + 1.5 * np.exp(-((f - 1000.0) ** 2) / (2 * 600.0 ** 2))
    return np.fft.irfft(spec * tilt * emph, n).astype(np.float32)


def add_noise(x: np.ndarray, snr_db: float, kind: str = "babble", seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n = _babble(len(x), rng) if kind == "babble" else rng.standard_normal(len(x)).astype(np.float32)
    p_sig = float(np.mean(x ** 2)) + 1e-12
    p_noi = float(np.mean(n ** 2)) + 1e-12
    scale = np.sqrt(p_sig / (p_noi * 10 ** (snr_db / 10.0)))
    y = x + scale * n
    peak = float(np.max(np.abs(y))) + 1e-9
    return (y / peak * 0.98).astype(np.float32) if peak > 1.0 else y.astype(np.float32)


def degrade(x: np.ndarray, condition: str, seed: int = 0) -> np.ndarray:
    """`clean` | `snr<db>` e.g. snr10, snr5, snr0, snr-5."""
    if condition == "clean":
        return x
    if condition.startswith("snr"):
        return add_noise(x, float(condition[3:]), seed=seed)
    raise ValueError(f"unknown condition {condition}")
