"""
Input/Output utilities for timbre sequence analysis
"""
import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Any, Optional


def list_audio_files(in_dir: Path, include_glob: Optional[List[str]] = None) -> List[Path]:
    """
    List audio files in directory with optional glob filtering
    
    Args:
        in_dir: Input directory
        include_glob: List of glob patterns to include
    
    Returns:
        List of audio file paths
    """
    audio_extensions = {'.wav', '.flac', '.mp3', '.ogg', '.m4a', '.aiff', '.aif'}
    
    if include_glob:
        files = []
        for pattern in include_glob:
            files.extend(in_dir.rglob(pattern))
        # Filter by audio extensions
        files = [f for f in files if f.suffix.lower() in audio_extensions]
    else:
        files = [f for f in in_dir.rglob('*') if f.suffix.lower() in audio_extensions]
    
    # Remove duplicates and sort
    uniq = sorted(list(set(files)))
    return uniq


def resolve_input_files(in_dir: Optional[Path], files: Optional[List[Path]], 
                       include_glob: Optional[List[str]] = None) -> List[Path]:
    """
    Resolve input files from directory or file list
    
    Args:
        in_dir: Input directory
        files: List of specific files
        include_glob: Glob patterns for filtering
    
    Returns:
        List of resolved file paths
    """
    if files:
        return files
    elif in_dir:
        return list_audio_files(in_dir, include_glob)
    else:
        return []


def write_fine_analysis_summary(out_dir: Path, results: Dict[str, Any]) -> None:
    """
    Write timbre sequence analysis summary information
    
    Args:
        out_dir: Output directory
        results: Timbre sequence analysis results
    """
    summary = {
        'config': {
            'segment_duration': results['config'].segment_duration,
            'curve_smooth_window': results['config'].curve_smooth_window,
            'target_points': results['config'].target_points,
            'fusion_method': results['config'].fusion_method
        },
        'tracks': {
            name: {
                'num_segments': len(result['segments']),
                'total_duration': result['segment_times'][-1] if len(result['segment_times']) > 0 else 0,
                'feature_dimension': result['fused_features'].shape[1] if len(result['fused_features']) > 0 else 0
            }
            for name, result in results['track_results'].items()
        },
        'global': {
            'total_tracks': len(results['track_names']),
            'total_segments': sum(len(result['segments']) for result in results['track_results'].values()),
            'fused_feature_dimension': results['all_fused_features'].shape[1] if len(results['all_fused_features']) > 0 else 0
        }
    }
    
    with open(out_dir / "fine_analysis_summary.json", 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)


def write_fine_features_csv(out_dir: Path, results: Dict[str, Any]) -> None:
    """
    Write fused features to CSV file
    
    Args:
        out_dir: Output directory
        results: Timbre sequence analysis results
    """
    # Prepare data
    all_features = results['all_fused_features']
    track_names = results['track_names']
    
    if len(all_features) == 0:
        return
    
    # Create DataFrame
    feature_cols = [f'feature_{i}' for i in range(all_features.shape[1])]
    df_data = []
    
    feature_start_idx = 0
    
    for track_name in track_names:
        track_result = results['track_results'][track_name]
        num_segments = len(track_result['segments'])
        track_features = track_result['fused_features']  # Use individual track features
        
        for j in range(num_segments):
            row = {
                'track_name': track_name,
                'segment_index': j,
                'segment_time': track_result['segment_times'][j],
                **{col: track_features[j, k] for k, col in enumerate(feature_cols)}
            }
            df_data.append(row)
    
    df = pd.DataFrame(df_data)
    df.to_csv(out_dir / "fine_features.csv", index=False)


def write_fine_segments_info(out_dir: Path, results: Dict[str, Any]) -> None:
    """
    Write segment information
    
    Args:
        out_dir: Output directory
        results: Timbre sequence analysis results
    """
    segments_data = []
    
    for track_name, result in results['track_results'].items():
        for i, (start, end) in enumerate(result['segments']):
            segments_data.append({
                'track_name': track_name,
                'segment_index': i,
                'start_frame': start,
                'end_frame': end,
                'start_time': result['segment_times'][i],
                'duration': (end - start) / 22050  # Assuming 22050 sample rate
            })
    
    df = pd.DataFrame(segments_data)
    df.to_csv(out_dir / "fine_segments_info.csv", index=False)

