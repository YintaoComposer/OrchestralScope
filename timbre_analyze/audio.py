from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import numpy as np
import librosa


@dataclass
class AudioConfig:
    target_sr: int = 44100
    pre_emphasis: float = 0.97


def load_audio_mono(path: Path, sr: int) -> np.ndarray:
    y, _ = librosa.load(path, sr=sr, mono=True)
    return y


def apply_pre_emphasis(y: np.ndarray, coef: float) -> np.ndarray:
    if coef <= 0:
        return y
    emphasized = np.empty_like(y)
    emphasized[0] = y[0]
    emphasized[1:] = y[1:] - coef * y[:-1]
    return emphasized


def loudness_normalize_rms(y: np.ndarray, target_rms: float = 0.1, eps: float = 1e-8) -> np.ndarray:
    rms = np.sqrt(np.mean(y**2) + eps)
    if rms < eps:
        return y
    gain = target_rms / rms
    y_norm = y * gain
    # Soft limiting to prevent clipping
    peak = np.max(np.abs(y_norm))
    if peak > 1.0:
        y_norm = y_norm / peak
    return y_norm


def preprocess_file(path: Path, cfg: AudioConfig) -> Tuple[np.ndarray, int]:
    y = load_audio_mono(path, sr=cfg.target_sr)
    y = apply_pre_emphasis(y, cfg.pre_emphasis)
    y = loudness_normalize_rms(y)
    return y, cfg.target_sr


