from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import librosa


@dataclass
class FeatureConfig:
    frame_ms: float = 46.0
    hop_ms: float = 10.0
    n_mfcc: int = 20
    include_delta: bool = True
    include_delta2: bool = True
    use_chroma: bool = True
    use_specflux: bool = True


def ms_to_samples(ms: float, sr: int) -> int:
    return int(round(ms * 1e-3 * sr))


def compute_features(y: np.ndarray, sr: int, cfg: FeatureConfig) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    n_fft = int(2 ** np.ceil(np.log2(ms_to_samples(cfg.frame_ms, sr))))
    hop_length = ms_to_samples(cfg.hop_ms, sr)

    mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length, power=2.0)
    log_mel = librosa.power_to_db(mel_spec + 1e-10)
    mfcc = librosa.feature.mfcc(S=log_mel, sr=sr, n_mfcc=cfg.n_mfcc)
    feats = [mfcc]
    if cfg.include_delta:
        feats.append(librosa.feature.delta(mfcc))
    if cfg.include_delta2:
        feats.append(librosa.feature.delta(mfcc, order=2))

    if cfg.use_chroma:
        chroma = librosa.feature.chroma_stft(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length)
        feats.append(chroma)
    else:
        chroma = None

    if cfg.use_specflux:
        S = np.abs(librosa.stft(y=y, n_fft=n_fft, hop_length=hop_length))
        flux = np.sqrt(np.sum(np.diff(S, axis=1, prepend=S[:, :1])**2, axis=0, keepdims=True))
        feats.append(flux)
    else:
        flux = None

    F = np.vstack(feats).astype(np.float32)
    # CMVN by feature dimension
    mean = F.mean(axis=1, keepdims=True)
    std = F.std(axis=1, keepdims=True) + 1e-8
    F = (F - mean) / std

    by_name = {
        "mfcc": mfcc,
        "chroma": chroma if chroma is not None else np.empty((0, F.shape[1])),
        "scflux": flux if flux is not None else np.empty((0, F.shape[1])),
    }
    return F, by_name


