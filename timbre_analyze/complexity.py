from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict

import numpy as np
import librosa
from scipy.signal import savgol_filter, find_peaks, peak_widths


@dataclass
class ComplexityConfig:
    window_beats: int = 8
    smooth_method: str = "savgol"  # savgol|ma|none
    savgol_window: int = 7
    savgol_polyorder: int = 2
    ma_window: int = 5
    p_min: float = 0.1  # peak prominence threshold
    merge_boundary_ms: float = 0.0  # optional de-jitter: merge boundaries closer than this


def _moving_average(x: np.ndarray, w: int) -> np.ndarray:
    if w <= 1:
        return x
    pad = w // 2
    xpad = np.pad(x, (pad, pad), mode='edge')
    kernel = np.ones(w) / w
    y = np.convolve(xpad, kernel, mode='valid')
    return y


def track_beats(y: np.ndarray, sr: int, hop_length: int) -> Tuple[np.ndarray, np.ndarray]:
    tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr, hop_length=hop_length)
    beat_times = librosa.frames_to_time(beat_frames, sr=sr, hop_length=hop_length)
    return beat_frames, beat_times


def local_bpm(beat_times: np.ndarray, k: int = 4) -> np.ndarray:
    if len(beat_times) < 2:
        return np.array([])
    intervals = np.diff(beat_times)
    if len(intervals) < 1:
        return np.array([])
    ma = _moving_average(intervals, max(1, k))
    bpm = 60.0 / np.maximum(ma, 1e-6)
    # Align to beat indices (same length as beat_times, pad with last value at the end)
    bpm_full = np.concatenate([bpm, bpm[-1:]]) if len(bpm) > 0 else np.array([0.0])
    return bpm_full


def merge_close_boundaries(times: np.ndarray, min_separation_ms: float) -> np.ndarray:
    if min_separation_ms <= 0 or len(times) <= 1:
        return times
    min_sep = min_separation_ms / 1000.0
    kept = [times[0]]
    for t in times[1:]:
        if (t - kept[-1]) >= min_sep:
            kept.append(t)
    return np.array(kept)


def compute_local_complexity(beat_times: np.ndarray, boundary_times: np.ndarray, cfg: ComplexityConfig, labels: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    # Debounce boundaries by time
    boundary_times = np.sort(boundary_times)
    boundary_times = merge_close_boundaries(boundary_times, cfg.merge_boundary_ms)
    B = cfg.window_beats
    C_raw = []
    beat_indices = np.arange(len(beat_times))
    for b in beat_indices:
        # Window center at beat b, covering B beats (B//2 on each side)
        left = max(0, b - B // 2)
        right = min(len(beat_times) - 1, b + B // 2 + 1)  # +1 to ensure covering B beat intervals
        t0 = beat_times[left]
        t1 = beat_times[right] if right < len(beat_times) else beat_times[-1]
        if t1 <= t0:
            C_raw.append(0.0)
            continue
        # Change density: count label switches within window
        if labels is not None and len(labels) > 0:
            # Map beat indices to label indices (assuming labels are aligned with beats)
            # If number of labels is less than beats, need interpolation or repetition
            if len(labels) < len(beat_times):
                # Insufficient labels, map proportionally
                label_indices = np.linspace(0, len(labels)-1, len(beat_times), dtype=int)
                mapped_labels = labels[label_indices]
            else:
                # Sufficient labels, directly take corresponding beats
                mapped_labels = labels[:len(beat_times)]
            
            # Get labels within window
            window_labels = []
            for beat_idx in range(left, min(right + 1, len(mapped_labels))):
                if beat_idx < len(mapped_labels):
                    window_labels.append(mapped_labels[beat_idx])
            
            # Calculate label switch count
            if len(window_labels) > 1:
                changes = sum(1 for i in range(1, len(window_labels)) if window_labels[i] != window_labels[i-1])
            else:
                changes = 0
            count = changes
        else:
            # Fallback to boundary count statistics
            count = np.sum((boundary_times >= t0) & (boundary_times < t1))
        # Normalize by actual beats covered (avoid bias from window cropping)
        beats_in_window = max(1, right - left)
        C_raw.append(float(count) / float(beats_in_window))
    C_raw = np.asarray(C_raw, dtype=float)
    # Smoothing
    if cfg.smooth_method == "savgol" and len(C_raw) >= cfg.savgol_window:
        C_smooth = savgol_filter(C_raw, window_length=cfg.savgol_window, polyorder=cfg.savgol_polyorder, mode='interp')
    elif cfg.smooth_method == "ma":
        C_smooth = _moving_average(C_raw, cfg.ma_window)
    else:
        C_smooth = C_raw.copy()
    return C_raw, C_smooth, beat_indices


def find_max_complexity_peak(C_smooth: np.ndarray, beat_indices: np.ndarray, beat_times: np.ndarray, p_min: float = 0.1) -> Optional[Dict]:
    if len(C_smooth) == 0:
        return None
    peaks, props = find_peaks(C_smooth, prominence=p_min)
    if len(peaks) == 0:
        return None
    # Take the one with maximum prominence; if tied, by peak value, then by earliest (np.lexsort primary key is the last)
    peak_vals = C_smooth[peaks]
    prominences = props.get('prominences', np.zeros_like(peak_vals))
    keys = (peaks, -peak_vals, -prominences)  # Primary key: prominence descending
    order = np.lexsort(keys)
    idx = peaks[order[0]] if len(order) > 0 else peaks[0]
    peak = int(idx)
    peak_C = float(C_smooth[peak])
    prominence = float(prominences[np.where(peaks == peak)[0][0]]) if len(prominences) > 0 else 0.0
    # Get full width at half maximum (FWHM)
    try:
        widths, _, _, _ = peak_widths(C_smooth, peaks, rel_height=0.5)
        width = float(widths[np.where(peaks == peak)[0][0]]) if len(widths) > 0 else 0.0
    except Exception:
        width = 0.0
    total_beats = max(1, len(beat_times))
    # Relative position: first beat=0, last beat=1
    denom = max(1, total_beats - 1)
    peak_r = float(peak) / float(denom)
    result = {
        'peak_beat': peak,
        'peak_r': peak_r,
        'peak_time': float(beat_times[min(peak, len(beat_times) - 1)]),
        'peak_C': peak_C,
        'prominence': prominence,
        'width_beats': width,
    }
    return result


def track_complexity(boundary_times: np.ndarray, beat_times: np.ndarray, cfg: ComplexityConfig, labels: Optional[np.ndarray] = None) -> Dict:
    C_raw, C_smooth, beat_idx = compute_local_complexity(beat_times, boundary_times, cfg, labels)
    peak = find_max_complexity_peak(C_smooth, beat_idx, beat_times, p_min=cfg.p_min)
    # Full track complexity (per beat): change density
    if labels is not None and len(labels) > 1:
        # Count total label switches in full track
        total_changes = sum(1 for i in range(1, len(labels)) if labels[i] != labels[i-1])
        C_track = float(total_changes) / float(max(1, len(labels) - 1))  # Change density
    else:
        # Fallback to boundary count statistics
        N_total = len(boundary_times)
        B_total = max(1, len(beat_times))
        C_track = float(N_total) / float(B_total)
    # Changes per second (additional)
    T_total = (beat_times[-1] - beat_times[0]) if len(beat_times) >= 2 else 0.0
    if labels is not None and len(labels) > 1:
        total_changes = sum(1 for i in range(1, len(labels)) if labels[i] != labels[i-1])
        C_time = float(total_changes) / max(1e-6, T_total) if T_total > 0 else float(total_changes)
    else:
        N_total = len(boundary_times)
        C_time = float(N_total) / max(1e-6, T_total) if T_total > 0 else float(N_total)
    return {
        'C_raw': C_raw,
        'C_smooth': C_smooth,
        'beat_times': beat_times,
        'peak': peak,
        'C_track': C_track,
        'C_time': C_time,
    }


