"""
Timbre Sequence Analysis Module
Based on fixed 0.5s segmentation and multi-feature fusion
"""
import numpy as np
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass
from pathlib import Path
import librosa
from scipy import signal
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


@dataclass
class FineAnalysisConfig:
    """Timbre sequence analysis configuration"""
    segment_duration: float = 0.5  # Fixed segment duration (seconds)
    curve_smooth_window: int = 5  # Curve smoothing window
    time_normalize_method: str = "interpolate"  # Time normalization method
    target_points: int = 100  # Target number of points
    fusion_method: str = "weighted"  # Feature fusion method
    feature_weights: Optional[Dict[str, float]] = None  # Feature weights


def fixed_segmentation(y: np.ndarray, sr: int, segment_duration: float) -> List[Tuple[int, int]]:
    """
    Fixed duration segmentation
    
    Args:
        y: Preprocessed audio signal
        sr: Sample rate
        segment_duration: Segment duration (seconds)
    
    Returns:
        List of segment boundaries [(start_frame, end_frame), ...]
    """
    hop_length = int(sr * segment_duration)
    total_frames = len(y)
    
    segments = []
    start = 0
    while start < total_frames:
        end = min(start + hop_length, total_frames)
        segments.append((start, end))
        start = end
    
    return segments


def extract_segment_features(y: np.ndarray, sr: int, segment: Tuple[int, int], 
                           frame_ms: float = 46.0, hop_ms: float = 10.0) -> Dict[str, np.ndarray]:
    """
    Extract multiple features for a single segment
    
    Args:
        y: Audio signal
        sr: Sample rate
        segment: Segment boundaries (start, end)
        frame_ms: Frame length (milliseconds)
        hop_ms: Hop length (milliseconds)
    
    Returns:
        Feature dictionary
    """
    start, end = segment
    y_seg = y[start:end]
    
    # Feature extraction logic
    n_fft = int(2 ** np.ceil(np.log2(ms_to_samples(frame_ms, sr))))
    hop_length = ms_to_samples(hop_ms, sr)
    
    # Ensure n_fft is not larger than segment length
    if n_fft > len(y_seg):
        n_fft = len(y_seg)
        # Ensure n_fft is still a power of 2
        n_fft = int(2 ** np.floor(np.log2(n_fft)))
    
    # MFCC features
    mel_spec = librosa.feature.melspectrogram(y=y_seg, sr=sr, n_fft=n_fft, hop_length=hop_length, power=2.0)
    log_mel = librosa.power_to_db(mel_spec + 1e-10)
    mfcc = librosa.feature.mfcc(S=log_mel, sr=sr, n_mfcc=13)
    
    # Spectral features
    spectral_centroid = librosa.feature.spectral_centroid(y=y_seg, sr=sr, hop_length=hop_length)
    spectral_rolloff = librosa.feature.spectral_rolloff(y=y_seg, sr=sr, hop_length=hop_length)
    spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y_seg, sr=sr, hop_length=hop_length)
    
    # Rhythm features removed - tempo detection unreliable for short segments
    
    # Zero crossing rate
    zcr = librosa.feature.zero_crossing_rate(y_seg, hop_length=hop_length)
    
    # RMS energy
    rms = librosa.feature.rms(y=y_seg, hop_length=hop_length)
    
    return {
        'mfcc': mfcc,
        'spectral_centroid': spectral_centroid,
        'spectral_rolloff': spectral_rolloff,
        'spectral_bandwidth': spectral_bandwidth,
        'zcr': zcr,
        'rms': rms,
        'segment_duration': np.array([(end - start) / sr])
    }


def ms_to_samples(ms: float, sr: int) -> int:
    """Convert milliseconds to samples"""
    return int(round(ms * 1e-3 * sr))


def preprocess_curves(features: Dict[str, np.ndarray], window: int = 5) -> Dict[str, np.ndarray]:
    """
    Curve preprocessing: smoothing and denoising
    
    Args:
        features: Original feature dictionary
        window: Smoothing window size
    
    Returns:
        Preprocessed feature dictionary
    """
    processed = {}
    
    for name, feat in features.items():
        if feat.ndim == 1:
            # 1D features, smooth directly
            if len(feat) > window:
                smoothed = signal.savgol_filter(feat, window, 2)
            else:
                smoothed = feat
        else:
            # Multi-dimensional features, smooth each dimension separately
            smoothed = np.zeros_like(feat)
            for i in range(feat.shape[0]):
                if feat.shape[1] > window:
                    smoothed[i] = signal.savgol_filter(feat[i], window, 2)
                else:
                    smoothed[i] = feat[i]
        
        processed[name] = smoothed
    
    return processed


def time_normalize_features(features: Dict[str, np.ndarray], target_points: int, 
                          method: str = "interpolate") -> Dict[str, np.ndarray]:
    """
    Time normalization: unify time dimensions of all features
    
    Args:
        features: Feature dictionary
        target_points: Target number of time points
        method: Normalization method
    
    Returns:
        Time-normalized feature dictionary
    """
    normalized = {}
    
    for name, feat in features.items():
        if feat.ndim == 1:
            # 1D features
            if len(feat) == target_points:
                normalized[name] = feat
            elif method == "interpolate":
                x_old = np.linspace(0, 1, len(feat))
                x_new = np.linspace(0, 1, target_points)
                normalized[name] = np.interp(x_new, x_old, feat)
            else:  # pad or truncate
                if len(feat) < target_points:
                    # Pad
                    pad_width = target_points - len(feat)
                    normalized[name] = np.pad(feat, (0, pad_width), mode='edge')
                else:
                    # Truncate
                    normalized[name] = feat[:target_points]
        else:
            # Multi-dimensional features
            if feat.shape[1] == target_points:
                normalized[name] = feat
            elif method == "interpolate":
                normalized_feat = np.zeros((feat.shape[0], target_points))
                for i in range(feat.shape[0]):
                    x_old = np.linspace(0, 1, feat.shape[1])
                    x_new = np.linspace(0, 1, target_points)
                    normalized_feat[i] = np.interp(x_new, x_old, feat[i])
                normalized[name] = normalized_feat
            else:  # pad or truncate
                if feat.shape[1] < target_points:
                    # Pad
                    pad_width = target_points - feat.shape[1]
                    normalized[name] = np.pad(feat, ((0, 0), (0, pad_width)), mode='edge')
                else:
                    # Truncate
                    normalized[name] = feat[:, :target_points]
    
    return normalized


def unify_point_counts(features: Dict[str, np.ndarray], target_points: int) -> Dict[str, np.ndarray]:
    """
    Unify point counts: ensure all features have the same number of points
    
    Args:
        features: Feature dictionary
        target_points: Target number of points
    
    Returns:
        Point-unified feature dictionary
    """
    return time_normalize_features(features, target_points, method="interpolate")


def fuse_features(features: Dict[str, np.ndarray], method: str = "weighted", 
                 weights: Optional[Dict[str, float]] = None) -> np.ndarray:
    """
    Feature fusion: combine multiple features into unified representation
    
    Args:
        features: Feature dictionary
        method: Fusion method
        weights: Feature weights
    
    Returns:
        Fused feature vector
    """
    if weights is None:
        # Weights based on information gain and feature contribution
        # User-defined weights based on empirical analysis
        weights = {
            'mfcc': 0.50,           # 50% - Main feature, multi-dimensional contribution, maximum information gain
            'spectral_centroid': 0.20,  # 20% - Timbre brightness, medium-high information gain
            'spectral_bandwidth': 0.10, # 10% - Timbre complexity, complementary to Centroid
            'zcr': 0.10,            # 10% - Helpful in speech/percussion identification, increased weight
            'spectral_rolloff': 0.06,   # 6% - Moderate contribution in some models, reduced weight
            'rms': 0.04             # 4% - Represents energy, complementary to other features, but weak alone, reduced weight
        }
    
    # Flatten all features
    flattened_features = []
    feature_names = []
    
    for name, feat in features.items():
        if name in weights:
            weight = weights[name]
            if feat.ndim == 1:
                flattened = feat.flatten()
            else:
                flattened = feat.flatten()
            
            # Standardize
            if len(flattened) > 0:
                scaler = StandardScaler()
                normalized = scaler.fit_transform(flattened.reshape(-1, 1)).flatten()
                weighted = normalized * weight
                flattened_features.append(weighted)
                feature_names.append(name)
    
    if not flattened_features:
        return np.array([])
    
    # Concatenate all features
    fused = np.concatenate(flattened_features)
    
    if method == "pca":
        # PCA dimensionality reduction
        if len(fused) > 50:  # Only use PCA when feature dimension is high
            pca = PCA(n_components=min(50, len(fused)))
            fused = pca.fit_transform(fused.reshape(1, -1)).flatten()
    
    return fused


def analyze_track_fine(y: np.ndarray, sr: int, cfg: FineAnalysisConfig) -> Dict:
    """
    Perform timbre sequence analysis on a single track
    
    Args:
        y: Preprocessed audio signal
        sr: Sample rate
        cfg: Timbre sequence analysis configuration
    
    Returns:
        Analysis result dictionary
    """
    # 1. Fixed segmentation
    segments = fixed_segmentation(y, sr, cfg.segment_duration)
    
    # 2. Feature extraction
    all_features = []
    segment_times = []
    
    for i, (start, end) in enumerate(segments):
        segment_time = start / sr
        segment_times.append(segment_time)
        
        features = extract_segment_features(y, sr, (start, end))
        all_features.append(features)
    
    # 3. Curve preprocessing
    processed_features = []
    for features in all_features:
        processed = preprocess_curves(features, cfg.curve_smooth_window)
        processed_features.append(processed)
    
    # 4. Time normalization
    normalized_features = []
    for features in processed_features:
        normalized = time_normalize_features(features, cfg.target_points, cfg.time_normalize_method)
        normalized_features.append(normalized)
    
    # 5. Point count unification
    unified_features = []
    for features in normalized_features:
        unified = unify_point_counts(features, cfg.target_points)
        unified_features.append(unified)
    
    # 6. Feature fusion
    fused_features = []
    for features in unified_features:
        fused = fuse_features(features, cfg.fusion_method, cfg.feature_weights)
        fused_features.append(fused)
    
    fused_features = np.array(fused_features)
    
    return {
        'segments': segments,
        'segment_times': np.array(segment_times),
        'raw_features': all_features,
        'processed_features': processed_features,
        'normalized_features': normalized_features,
        'unified_features': unified_features,
        'fused_features': fused_features,
        'config': cfg
    }


def analyze_multiple_tracks_fine(track_data: List[Tuple[str, np.ndarray, int]], 
                               cfg: FineAnalysisConfig) -> Dict:
    """
    Perform timbre sequence analysis on multiple tracks
    
    Args:
        track_data: Track data list [(name, y, sr), ...]
        cfg: Timbre sequence analysis configuration
    
    Returns:
        Multi-track analysis results
    """
    results = {}
    all_fused_features = []
    track_names = []
    
    for name, y, sr in track_data:
        print(f"[sequence-analysis] Analyzing track: {name}")
        result = analyze_track_fine(y, sr, cfg)
        results[name] = result
        all_fused_features.append(result['fused_features'])
        track_names.append(name)
    
    # Combine fused features from all tracks
    if all_fused_features:
        all_fused_features = np.vstack(all_fused_features)
    
    return {
        'track_results': results,
        'track_names': track_names,
        'all_fused_features': all_fused_features,
        'config': cfg
    }

