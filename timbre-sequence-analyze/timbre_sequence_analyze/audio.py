"""
Audio preprocessing module for timbre sequence analysis
"""
import numpy as np
from pathlib import Path
from dataclasses import dataclass
from typing import Tuple
import librosa


@dataclass
class AudioConfig:
    """Audio preprocessing configuration"""
    target_sr: int = 44100
    pre_emphasis: float = 0.97


def pre_emphasis_filter(y: np.ndarray, alpha: float = 0.97) -> np.ndarray:
    """
    Apply pre-emphasis filter to audio signal
    
    Args:
        y: Audio signal
        alpha: Pre-emphasis coefficient
    
    Returns:
        Pre-emphasized audio signal
    """
    if len(y) == 0:
        return y
    
    # Apply pre-emphasis: y[n] = y[n] - alpha * y[n-1]
    emphasized = np.zeros_like(y)
    emphasized[0] = y[0]
    for i in range(1, len(y)):
        emphasized[i] = y[i] - alpha * y[i-1]
    
    return emphasized


def loudness_normalize_rms(y: np.ndarray, target_rms: float = 0.1, eps: float = 1e-8) -> np.ndarray:
    """
    Normalize audio loudness using RMS
    
    Args:
        y: Audio signal
        target_rms: Target RMS value
        eps: Small value to avoid division by zero
    
    Returns:
        Loudness-normalized audio signal
    """
    current_rms = np.sqrt(np.mean(y**2))
    if current_rms < eps:
        return y
    
    y_norm = y * (target_rms / current_rms)
    return y_norm


def preprocess_file(path: Path, cfg: AudioConfig) -> Tuple[np.ndarray, int]:
    """
    Preprocess audio file for analysis
    
    Args:
        path: Path to audio file
        cfg: Audio configuration
    
    Returns:
        Tuple of (preprocessed_audio, sample_rate)
    """
    # Convert Path to string for librosa compatibility
    path_str = str(path)
    
    # Load audio
    y, sr = librosa.load(path_str, sr=cfg.target_sr, mono=True)
    
    # Apply pre-emphasis
    y = pre_emphasis_filter(y, cfg.pre_emphasis)
    
    # Normalize loudness
    y = loudness_normalize_rms(y)
    
    return y, cfg.target_sr

