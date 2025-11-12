from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import librosa


@dataclass
class SegmentationConfig:
    kernel_gauss: float = 3.0
    beat_sync: bool = True
    min_seg_beats: int = 2
    adaptive_segmentation: bool = True  # Enable adaptive segmentation
    min_seg_beats_traditional: int = 4  # Minimum segment length when timbre complexity is low (in beats)
    min_seg_beats_contemporary: int = 1  # Minimum segment length when timbre complexity is high (in beats)


def _detect_boundaries_from_novelty(y: np.ndarray, sr: int, hop_length: int) -> np.ndarray:
    nov = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop_length)
    peaks = librosa.util.peak_pick(nov, pre_max=16, post_max=16, pre_avg=16, post_avg=16, delta=np.median(nov)*0.5, wait=16)
    # Boundaries in frame indices
    bounds = np.unique(np.concatenate([[0], peaks, [len(nov) - 1]]))
    return bounds


def _align_bounds_to_beats(bounds: np.ndarray, beat_frames: np.ndarray, min_seg_beats: int) -> np.ndarray:
    if beat_frames.size == 0:
        return bounds
    # Snap boundaries to nearest beat
    aligned = []
    for b in bounds:
        idx = np.argmin(np.abs(beat_frames - b))
        aligned.append(int(beat_frames[idx]))
    aligned = np.unique(np.array(aligned, dtype=int))
    # Enforce minimum beat segment
    if aligned.size <= 2:
        return aligned
    kept = [aligned[0]]
    last_idx = 0
    for i in range(1, len(aligned)):
        if (i - last_idx) >= min_seg_beats:
            kept.append(aligned[i])
            last_idx = i
    if kept[-1] != aligned[-1]:
        kept[-1] = aligned[-1]
    return np.array(kept, dtype=int)


def _estimate_timbre_complexity(y: np.ndarray, sr: int, hop_length: int) -> float:
    """
    Estimate timbre complexity based on audio features, returns a score between 0-1
    Higher score indicates more frequent timbre changes, requiring finer segmentation
    """
    # 1. Spectral centroid variation rate (timbre brightness changes)
    spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr, hop_length=hop_length)[0]
    spectral_centroid_var = np.var(spectral_centroids)
    
    # 2. Spectral bandwidth variation rate (timbre richness changes)
    spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr, hop_length=hop_length)[0]
    spectral_bandwidth_var = np.var(spectral_bandwidth)
    
    # 3. Zero crossing rate variation (timbre texture changes)
    zcr = librosa.feature.zero_crossing_rate(y, hop_length=hop_length)[0]
    zcr_var = np.var(zcr)
    
    # 4. MFCC variation complexity (timbre feature changes)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, hop_length=hop_length, n_mfcc=13)
    mfcc_var = np.mean(np.var(mfcc, axis=1))
    
    # 5. Spectral rolloff point variation (high-frequency content changes)
    spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr, hop_length=hop_length)[0]
    spectral_rolloff_var = np.var(spectral_rolloff)
    
    # 6. Chroma feature variation (harmonic color changes)
    chroma = librosa.feature.chroma_stft(y=y, sr=sr, hop_length=hop_length)
    chroma_var = np.mean(np.var(chroma, axis=1))
    
    # Normalize each metric to 0-1 range
    def normalize_feature(feat_var, typical_range=(0, 1)):
        return np.clip(feat_var / typical_range[1], 0, 1)
    
    # Comprehensive timbre complexity score
    timbre_complexity = (
        normalize_feature(spectral_centroid_var, (0, 1000)) * 0.2 +
        normalize_feature(spectral_bandwidth_var, (0, 1000)) * 0.15 +
        normalize_feature(zcr_var, (0, 0.1)) * 0.15 +
        normalize_feature(mfcc_var, (0, 10)) * 0.25 +
        normalize_feature(spectral_rolloff_var, (0, 1000)) * 0.15 +
        normalize_feature(chroma_var, (0, 0.1)) * 0.1
    )
    
    return np.clip(timbre_complexity, 0, 1)


def segment_audio(y: np.ndarray, sr: int, frame_ms: float, hop_ms: float, cfg: SegmentationConfig) -> Tuple[np.ndarray, np.ndarray]:
    hop_length = int(round(hop_ms * 1e-3 * sr))
    bounds = _detect_boundaries_from_novelty(y, sr, hop_length)
    
    # Adaptive segmentation: dynamically adjust min_seg_beats based on timbre complexity
    if cfg.adaptive_segmentation:
        timbre_complexity = _estimate_timbre_complexity(y, sr, hop_length)
        
        # Calculate min_seg_beats by linear interpolation based on timbre complexity
        # High complexity (1.0) -> finer segmentation (contemporary)
        # Low complexity (0.0) -> coarser segmentation (traditional)
        min_seg_beats = int(
            cfg.min_seg_beats_traditional * (1 - timbre_complexity) + 
            cfg.min_seg_beats_contemporary * timbre_complexity
        )
        
        # Ensure within reasonable range
        min_seg_beats = max(1, min(min_seg_beats, 8))
        
        # Optional: print debug information
        # print(f"Timbre complexity: {timbre_complexity:.3f}, adjusted min_seg_beats: {min_seg_beats}")
    else:
        min_seg_beats = cfg.min_seg_beats
    
    if cfg.beat_sync:
        try:
            tempo, beats = librosa.beat.beat_track(y=y, sr=sr, hop_length=hop_length)
        except AttributeError:
            # Fallback: use onset detection for beat tracking
            onset_frames = librosa.onset.onset_detect(y=y, sr=sr, hop_length=hop_length)
            tempo = 120.0  # Default tempo
            beats = onset_frames
        bounds = _align_bounds_to_beats(bounds, beats, min_seg_beats)
    # Convert to time (seconds)
    times = librosa.frames_to_time(bounds, sr=sr, hop_length=hop_length)
    return bounds, times


def frames_to_segments(bounds: np.ndarray, num_frames: int) -> List[Tuple[int, int]]:
    segs: List[Tuple[int, int]] = []
    for i in range(len(bounds) - 1):
        a = int(bounds[i])
        b = int(bounds[i + 1])
        a = max(0, min(a, num_frames - 1))
        b = max(a + 1, min(b, num_frames))
        segs.append((a, b))
    if bounds[-1] < num_frames - 1:
        segs.append((int(bounds[-1]), num_frames))
    return segs


