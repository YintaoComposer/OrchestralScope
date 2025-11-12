"""
Visualization module for timbre sequence analysis
"""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Any, Optional
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler


def plot_feature_trends(results: Dict[str, Any], output_dir: Path, 
                       feature_names: Optional[List[str]] = None) -> None:
    """
    Plot feature trends for multiple tracks
    
    Args:
        results: Fine analysis results
        output_dir: Output directory
        feature_names: List of feature names to plot (if None, plot all)
    """
    if feature_names is None:
        feature_names = ['mfcc', 'spectral_centroid', 'spectral_rolloff', 
                        'spectral_bandwidth', 'zcr', 'rms']
    
    track_results = results['track_results']
    track_names = results['track_names']
    
    # Set up the plot style
    plt.style.use('default')
    sns.set_palette("husl")
    
    # Create individual trend plots for each track
    for track_name in track_names:
        track_result = track_results[track_name]
        segment_times = track_result['segment_times']
        
        # Create subplots for each feature for this track
        n_features = len(feature_names)
        fig, axes = plt.subplots(n_features, 1, figsize=(14, 3 * n_features))
        if n_features == 1:
            axes = [axes]
        
        for i, feature_name in enumerate(feature_names):
            ax = axes[i]
            
            # Get feature data for this track
            feature_values = []
            for seg_features in track_result['raw_features']:
                if feature_name in seg_features:
                    feat = seg_features[feature_name]
                    if feat.ndim > 1:
                        # For multi-dimensional features, take mean
                        feature_values.append(np.mean(feat))
                    else:
                        feature_values.append(feat[0] if len(feat) > 0 else 0)
                else:
                    feature_values.append(0)
            
            if feature_values and len(feature_values) == len(segment_times):
                # Plot the trend with clear markers and grid
                ax.plot(segment_times, feature_values, 
                       color='blue', linewidth=2.5, alpha=0.8, marker='o', 
                       markersize=4, markerfacecolor='red', markeredgecolor='darkred')
                
                # Add value annotations for key points
                if len(feature_values) > 0:
                    # Annotate min and max values
                    min_idx = np.argmin(feature_values)
                    max_idx = np.argmax(feature_values)
                    
                    ax.annotate(f'Min: {feature_values[min_idx]:.3f}', 
                              xy=(segment_times[min_idx], feature_values[min_idx]),
                              xytext=(10, 10), textcoords='offset points',
                              bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7),
                              arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
                    
                    ax.annotate(f'Max: {feature_values[max_idx]:.3f}', 
                              xy=(segment_times[max_idx], feature_values[max_idx]),
                              xytext=(10, -20), textcoords='offset points',
                              bbox=dict(boxstyle='round,pad=0.3', facecolor='lightgreen', alpha=0.7),
                              arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
            
            # Enhanced axis formatting
            ax.set_title(f'{feature_name.replace("_", " ").title()} Trend', 
                        fontsize=14, fontweight='bold', pad=15)
            ax.set_xlabel('Time (seconds)', fontsize=12, fontweight='bold')
            ax.set_ylabel(f'{feature_name.replace("_", " ").title()}', fontsize=12, fontweight='bold')
            
            # Add grid with major and minor ticks
            ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
            ax.grid(True, alpha=0.1, linestyle=':', linewidth=0.3, which='minor')
            ax.minorticks_on()
            
            # Format x-axis to show time clearly
            ax.tick_params(axis='x', labelsize=10, rotation=45)
            ax.tick_params(axis='y', labelsize=10)
            
            # Add statistics text box
            if feature_values:
                stats_text = f'Mean: {np.mean(feature_values):.3f}\nStd: {np.std(feature_values):.3f}\nRange: {np.max(feature_values)-np.min(feature_values):.3f}'
                ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
                       verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
                       fontsize=9)
        
        # Clean track name for filename
        clean_name = "".join(c for c in track_name if c.isalnum() or c in (' ', '-', '_')).rstrip()
        clean_name = clean_name.replace(' ', '_')[:50]  # Limit filename length
        
        plt.suptitle(f'Feature Trends: {track_name}', fontsize=16, fontweight='bold', y=0.98)
        plt.tight_layout()
        plt.subplots_adjust(top=0.95)
        plt.savefig(output_dir / f'{clean_name}_feature_trends.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Feature trends plot saved: {output_dir / f'{clean_name}_feature_trends.png'}")
    
    # Also create a combined comparison plot
    n_features = len(feature_names)
    fig, axes = plt.subplots(n_features, 1, figsize=(14, 3 * n_features))
    if n_features == 1:
        axes = [axes]
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(track_names)))
    
    for i, feature_name in enumerate(feature_names):
        ax = axes[i]
        
        for j, track_name in enumerate(track_names):
            track_result = track_results[track_name]
            segment_times = track_result['segment_times']
            
            # Get feature data for this track
            feature_values = []
            for seg_features in track_result['raw_features']:
                if feature_name in seg_features:
                    feat = seg_features[feature_name]
                    if feat.ndim > 1:
                        # For multi-dimensional features, take mean
                        feature_values.append(np.mean(feat))
                    else:
                        feature_values.append(feat[0] if len(feat) > 0 else 0)
                else:
                    feature_values.append(0)
            
            if feature_values and len(feature_values) == len(segment_times):
                # Plot the trend
                ax.plot(segment_times, feature_values, 
                       label=track_name[:20] + '...' if len(track_name) > 20 else track_name,
                       color=colors[j], linewidth=2, alpha=0.8, marker='o', markersize=3)
        
        ax.set_title(f'{feature_name.replace("_", " ").title()} Trends (All Tracks)', 
                    fontsize=14, fontweight='bold', pad=15)
        ax.set_xlabel('Time (seconds)', fontsize=12, fontweight='bold')
        ax.set_ylabel(f'{feature_name.replace("_", " ").title()}', fontsize=12, fontweight='bold')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.tick_params(axis='x', labelsize=10, rotation=45)
        ax.tick_params(axis='y', labelsize=10)
    
    plt.suptitle('Feature Trends Comparison Across All Tracks', fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout()
    plt.subplots_adjust(top=0.95)
    plt.savefig(output_dir / 'all_tracks_feature_trends.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Combined feature trends plot saved: {output_dir / 'all_tracks_feature_trends.png'}")


def plot_hierarchical_clustering(results: Dict[str, Any], output_dir: Path) -> None:
    """
    Plot hierarchical clustering dendrogram between tracks
    
    Args:
        results: Fine analysis results
        output_dir: Output directory
    """
    from scipy.cluster.hierarchy import dendrogram, linkage
    from scipy.spatial.distance import pdist
    
    track_names = results['track_names']
    track_results = results['track_results']
    
    if len(track_names) < 2:
        print("⚠️ Need at least 2 tracks for hierarchical clustering")
        return
    
    # Calculate average features for each track
    track_avg_features = []
    for track_name in track_names:
        track_result = track_results[track_name]
        # Use the fused features and take mean across segments
        fused_features = track_result['fused_features']
        avg_features = np.mean(fused_features, axis=0)
        track_avg_features.append(avg_features)
    
    track_avg_features = np.array(track_avg_features)
    
    # Standardize features
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(track_avg_features)
    
    # Calculate distance matrix
    distance_matrix = pdist(features_scaled, metric='cosine')
    
    # Perform hierarchical clustering
    linkage_matrix = linkage(distance_matrix, method='average')
    
    # Create dendrogram
    plt.figure(figsize=(12, 8))
    
    # Keep only the first word of track names (separated by space or underscore)
    try:
        import re
        labels_use = []
        for n in track_names:
            tok = re.split(r"[ _]+", str(n).strip())
            first = tok[0] if tok and tok[0] else str(n)
            labels_use.append(first)
    except Exception:
        labels_use = track_names
    
    dendrogram(linkage_matrix, 
               labels=labels_use,
               orientation='top',
               distance_sort='descending',
               show_leaf_counts=True)
    
    plt.title('Hierarchical Clustering of Tracks\n(Based on Fused Features)', 
              fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('Tracks', fontsize=12)
    plt.ylabel('Distance', fontsize=12)
    # Labels upright, smaller font
    plt.xticks(rotation=0, ha='center', fontsize=5)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'hierarchical_clustering.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Save clustering data to CSV
    write_hierarchical_clustering_csv(track_names, track_avg_features, distance_matrix, linkage_matrix, output_dir)
    
    print(f"✓ Hierarchical clustering plot saved: {output_dir / 'hierarchical_clustering.png'}")
    print(f"✓ Hierarchical clustering data saved: {output_dir / 'hierarchical_clustering_data.csv'}")


def write_hierarchical_clustering_csv(track_names: List[str], track_avg_features: np.ndarray, distance_matrix: np.ndarray, 
                                     linkage_matrix: np.ndarray, output_dir: Path) -> None:
    """
    Write hierarchical clustering data to CSV files
    
    Args:
        track_names: List of track names
        track_avg_features: Average fused features per track (n_tracks, n_features)
        distance_matrix: Pairwise distance matrix
        linkage_matrix: Linkage matrix from hierarchical clustering
        output_dir: Output directory
    """
    from scipy.spatial.distance import squareform
    
    # 1. Distance matrix CSV
    distance_df = pd.DataFrame(
        squareform(distance_matrix), 
        index=track_names, 
        columns=track_names
    )
    distance_df.to_csv(output_dir / 'hierarchical_clustering_distances.csv')
    
    # 2. Linkage matrix CSV (clustering steps)
    linkage_df = pd.DataFrame(
        linkage_matrix,
        columns=['cluster1', 'cluster2', 'distance', 'count']
    )
    linkage_df.to_csv(output_dir / 'hierarchical_clustering_steps.csv', index=False)
    
    # 3. Summary CSV with cluster information
    from scipy.cluster.hierarchy import fcluster
    
    # Get clusters at different distance thresholds
    thresholds = [0.5, 1.0, 1.5, 2.0]
    cluster_summary = []
    
    for threshold in thresholds:
        clusters = fcluster(linkage_matrix, threshold, criterion='distance')
        unique_clusters = np.unique(clusters)
        
        for cluster_id in unique_clusters:
            track_indices = np.where(clusters == cluster_id)[0]
            cluster_tracks = [track_names[i] for i in track_indices]
            cluster_summary.append({
                'threshold': threshold,
                'cluster_id': cluster_id,
                'cluster_size': len(cluster_tracks),
                'tracks': '; '.join(cluster_tracks)
            })
    
    cluster_df = pd.DataFrame(cluster_summary)
    cluster_df.to_csv(output_dir / 'hierarchical_clustering_summary.csv', index=False)
    
    # 4. Detailed cluster similarity analysis
    write_cluster_similarity_analysis(track_names, track_avg_features, linkage_matrix, output_dir)
    
    print(f"✓ Distance matrix saved: {output_dir / 'hierarchical_clustering_distances.csv'}")
    print(f"✓ Clustering steps saved: {output_dir / 'hierarchical_clustering_steps.csv'}")
    print(f"✓ Cluster summary saved: {output_dir / 'hierarchical_clustering_summary.csv'}")
    print(f"✓ Cluster similarity analysis saved: {output_dir / 'cluster_similarity_analysis.csv'}")


def write_cluster_similarity_analysis(track_names: List[str], track_avg_features: np.ndarray, 
                                    linkage_matrix: np.ndarray, output_dir: Path) -> None:
    """
    Analyze and write detailed similarity features for each cluster
    
    Args:
        track_names: List of track names
        track_avg_features: Average features for each track
        linkage_matrix: Linkage matrix from hierarchical clustering
        output_dir: Output directory
    """
    from scipy.cluster.hierarchy import fcluster
    
    # Feature names (based on the 5 key features used in fine analysis, tempo removed)
    feature_names = [
        'mfcc_mean', 'spectral_centroid', 'spectral_rolloff', 
        'zcr', 'rms'
    ]
    
    # Get clusters at different thresholds
    thresholds = [0.5, 1.0, 1.5, 2.0]
    analysis_results = []
    
    for threshold in thresholds:
        clusters = fcluster(linkage_matrix, threshold, criterion='distance')
        unique_clusters = np.unique(clusters)
        
        for cluster_id in unique_clusters:
            track_indices = np.where(clusters == cluster_id)[0]
            cluster_tracks = [track_names[i] for i in track_indices]
            cluster_features = track_avg_features[track_indices]
            
            if len(cluster_tracks) >= 2:  # Only analyze clusters with 2+ tracks
                # Calculate similarity metrics within cluster
                cluster_analysis = analyze_cluster_similarity(
                    cluster_tracks, cluster_features, feature_names, 
                    threshold, cluster_id
                )
                analysis_results.extend(cluster_analysis)
    
    # Save to CSV
    if analysis_results:
        analysis_df = pd.DataFrame(analysis_results)
        analysis_df.to_csv(output_dir / 'cluster_similarity_analysis.csv', index=False)
    else:
        print("⚠️ No clusters with 2+ tracks found for similarity analysis")


def analyze_cluster_similarity(cluster_tracks: List[str], cluster_features: np.ndarray, 
                             feature_names: List[str], threshold: float, cluster_id: int) -> List[Dict]:
    """
    Analyze similarity within a cluster
    
    Args:
        cluster_tracks: Track names in the cluster
        cluster_features: Feature matrix for tracks in cluster
        feature_names: Names of features
        threshold: Distance threshold used
        cluster_id: Cluster identifier
        
    Returns:
        List of analysis results
    """
    results = []
    n_tracks = len(cluster_tracks)
    
    # Calculate pairwise similarities within cluster
    for i in range(n_tracks):
        for j in range(i + 1, n_tracks):
            track1, track2 = cluster_tracks[i], cluster_tracks[j]
            features1, features2 = cluster_features[i], cluster_features[j]
            
            # Calculate feature-wise similarities
            feature_similarities = []
            for k, feature_name in enumerate(feature_names):
                val1, val2 = features1[k], features2[k]
                # Use normalized difference similarity
                similarity = 1.0 - abs(val1 - val2) / max(abs(val1), abs(val2), 1e-8)
                feature_similarities.append(similarity)
            
            # Overall similarity (cosine similarity)
            from sklearn.metrics.pairwise import cosine_similarity
            overall_sim = cosine_similarity([features1], [features2])[0][0]
            
            # Find most similar features (top 3)
            feature_sim_pairs = list(zip(feature_names, feature_similarities))
            feature_sim_pairs.sort(key=lambda x: x[1], reverse=True)
            top_similar_features = feature_sim_pairs[:3]
            
            # Find least similar features (bottom 2)
            least_similar_features = feature_sim_pairs[-2:]
            
            results.append({
                'threshold': threshold,
                'cluster_id': cluster_id,
                'cluster_size': n_tracks,
                'track1': track1,
                'track2': track2,
                'overall_similarity': overall_sim,
                'mfcc_similarity': feature_similarities[0],
                'spectral_centroid_similarity': feature_similarities[1],
                'spectral_rolloff_similarity': feature_similarities[2],
                'zcr_similarity': feature_similarities[3],
                'rms_similarity': feature_similarities[4],
                'most_similar_feature1': top_similar_features[0][0],
                'most_similar_value1': top_similar_features[0][1],
                'most_similar_feature2': top_similar_features[1][0],
                'most_similar_value2': top_similar_features[1][1],
                'most_similar_feature3': top_similar_features[2][0],
                'most_similar_value3': top_similar_features[2][1],
                'least_similar_feature1': least_similar_features[0][0],
                'least_similar_value1': least_similar_features[0][1],
                'least_similar_feature2': least_similar_features[1][0],
                'least_similar_value2': least_similar_features[1][1],
                'cluster_tracks': '; '.join(cluster_tracks)
            })
    
    return results


def calculate_detailed_similarities(results: Dict[str, Any], output_dir: Path, use_dtw: bool = False) -> None:
    """
    Calculate detailed similarity metrics for each track and save to CSV
    Each track gets one row, sorted by similarity, with least similar pair at first and last positions
    
    Args:
        results: Fine analysis results
        output_dir: Output directory
    """
    track_names = results['track_names']
    track_results = results['track_results']
    
    if len(track_names) < 2:
        print("⚠️ Need at least 2 tracks for similarity calculation")
        return
    
    # Define the 5 key features (tempo removed)
    key_features = ['mfcc', 'spectral_centroid', 'spectral_rolloff', 
                   'spectral_bandwidth', 'zcr', 'rms']
    
    # Collect ALL feature values for each track (not just means) - using all time points
    track_all_features = {}
    for track_name in track_names:
        track_result = track_results[track_name]
        track_all_features[track_name] = {}
        
        for feature_name in key_features:
            all_feature_values = []
            for seg_features in track_result['raw_features']:
                if feature_name in seg_features:
                    feat = seg_features[feature_name]
                    if feat.ndim > 1:
                        if feature_name == 'mfcc':
                            # For MFCC, use first 5 coefficients across all time points
                            # Shape: (13, time_frames) -> (5, time_frames)
                            mfcc_subset = feat[:5, :]  # First 5 coefficients
                            # Flatten to get all time points: (5 * time_frames,)
                            all_feature_values.extend(mfcc_subset.flatten())
                        else:
                            # For other features, flatten all time points
                            all_feature_values.extend(feat.flatten())
                    else:
                        # 1D features, add all values
                        all_feature_values.extend(feat)
            
            track_all_features[track_name][feature_name] = np.array(all_feature_values)
    
    # Handle different feature lengths by padding or truncating to same length
    # Find the minimum length for each feature across all tracks
    feature_min_lengths = {}
    for feature_name in key_features:
        lengths = [len(track_all_features[track][feature_name]) for track in track_names 
                  if len(track_all_features[track][feature_name]) > 0]
        feature_min_lengths[feature_name] = min(lengths) if lengths else 0
    
    # Create standardized feature vectors for each track
    track_feature_vectors = {}
    for track_name in track_names:
        track_vector = []
        for feature_name in key_features:
            feature_data = track_all_features[track_name][feature_name]
            if len(feature_data) > 0:
                # Truncate or pad to minimum length
                min_len = feature_min_lengths[feature_name]
                if len(feature_data) > min_len:
                    # Truncate to minimum length
                    feature_data = feature_data[:min_len]
                elif len(feature_data) < min_len:
                    # Pad with zeros
                    feature_data = np.pad(feature_data, (0, min_len - len(feature_data)), 'constant')
                track_vector.extend(feature_data)
            else:
                # If no data, pad with zeros
                track_vector.extend([0.0] * feature_min_lengths[feature_name])
        
        track_feature_vectors[track_name] = np.array(track_vector)
    
    # Get number of tracks
    n_tracks = len(track_names)
    
    # Compute pairwise similarities incrementally to reduce memory
    # Also downsample very long aligned sequences to cap memory/time
    def _pair_overall_similarity(name_i: str, name_j: str) -> float:
        concatenated_i: list = []
        concatenated_j: list = []
        max_points_per_feature: int = 50000  # cap after DTW alignment per feature
        for feature_name in key_features:
            fi = track_all_features[name_i][feature_name]
            fj = track_all_features[name_j][feature_name]
            if len(fi) == 0 or len(fj) == 0:
                continue
            ai, aj = _dtw_align_1d(fi.astype(np.float32, copy=False), fj.astype(np.float32, copy=False))
            # Downsample if too long
            if len(ai) > max_points_per_feature:
                step = int(np.ceil(len(ai) / max_points_per_feature))
                ai = ai[::step]
                aj = aj[::step]
            concatenated_i.append(ai)
            concatenated_j.append(aj)
        if len(concatenated_i) == 0:
            return 0.0
        vi = np.concatenate(concatenated_i).astype(np.float32, copy=False)
        vj = np.concatenate(concatenated_j).astype(np.float32, copy=False)
        if np.all(vi == 0) and np.all(vj == 0):
            return 1.0
        dot_product = float(np.dot(vi, vj))
        norm_i = float(np.linalg.norm(vi))
        norm_j = float(np.linalg.norm(vj))
        sim = 0.0 if (norm_i == 0.0 or norm_j == 0.0) else dot_product / (norm_i * norm_j)
        return float(max(0.0, min(1.0, sim)))
    
    similarity_matrix = np.zeros((n_tracks, n_tracks), dtype=np.float32)
    for i in range(n_tracks):
        for j in range(i + 1, n_tracks):
            sim_ij = _pair_overall_similarity(track_names[i], track_names[j])
            similarity_matrix[i, j] = sim_ij
            similarity_matrix[j, i] = sim_ij
    
    # Helper: DTW-based alignment returning aligned versions of two 1D arrays
    def _dtw_align_1d(a: np.ndarray, b: np.ndarray) -> (np.ndarray, np.ndarray):
        if not use_dtw:
            m = min(len(a), len(b))
            return a[:m], b[:m]
        try:
            from librosa.sequence import dtw
        except Exception:
            # Fallback to simple trimming if librosa dtw not available
            m = min(len(a), len(b))
            return a[:m], b[:m]
        # Compute cost matrix on z-normalized sequences for stability
        a_z = (a - np.mean(a)) / (np.std(a) + 1e-8)
        b_z = (b - np.mean(b)) / (np.std(b) + 1e-8)
        # Use squared euclidean
        import numpy as _np
        D = _np.square(_np.subtract.outer(a_z, b_z))
        _, wp = dtw(C=D)
        wp = wp[::-1]
        a_aligned = _np.array([a[i] for i, _ in wp], dtype=float)
        b_aligned = _np.array([b[j] for _, j in wp], dtype=float)
        return a_aligned, b_aligned

    # Calculate average similarity for each track (with all other tracks)
    # Use the same method as individual features for consistency
    track_avg_similarities = []
    for i, track_name in enumerate(track_names):
        # Calculate overall similarity using the same method as individual features
        overall_similarities = []
        for k in range(n_tracks):
            if k != i:  # Don't compare with itself
                # Use the same feature vectors as individual calculation
                # Reconstruct concatenated vector per feature with optional DTW alignment
                concatenated_i: list = []
                concatenated_k: list = []
                for feature_name in key_features:
                    fi = track_all_features[track_name][feature_name]
                    fk = track_all_features[track_names[k]][feature_name]
                    if len(fi) == 0 or len(fk) == 0:
                        continue
                    ai, ak = _dtw_align_1d(fi, fk)
                    concatenated_i.append(ai)
                    concatenated_k.append(ak)
                if len(concatenated_i) == 0:
                    sim = 0.0
                else:
                    vi = np.concatenate(concatenated_i)
                    vk = np.concatenate(concatenated_k)
                    if np.all(vi == 0) and np.all(vk == 0):
                        sim = 1.0
                    else:
                        dot_product = np.dot(vi, vk)
                        norm_i = np.linalg.norm(vi)
                        norm_k = np.linalg.norm(vk)
                        sim = 0.0 if (norm_i == 0 or norm_k == 0) else dot_product / (norm_i * norm_k)
                        sim = max(0, min(1, sim))
                
                overall_similarities.append(sim)
        
        avg_sim = np.mean(overall_similarities) if overall_similarities else 0.0
        track_avg_similarities.append(avg_sim)
    
    # Find the least similar pair
    with np.errstate(invalid='ignore'):
        masked = similarity_matrix + np.eye(n_tracks, dtype=np.float32) * np.float32(2.0)
        flat_idx = int(np.argmin(masked))
    least_similar_pair = (flat_idx // n_tracks, flat_idx % n_tracks)
    min_similarity = float(similarity_matrix[least_similar_pair])
    
    # Create track data with individual feature similarities
    track_data = []
    for i, track_name in enumerate(track_names):
        row = {'track_name': track_name}
        
        # Calculate individual feature similarities using all time points
        for feature_name in key_features:
            feature_similarities = []
            for k in range(n_tracks):
                if k != i:  # Don't compare with itself
                    # Extract feature data for both tracks
                    feat_i = track_all_features[track_name][feature_name]
                    feat_k = track_all_features[track_names[k]][feature_name]
                    
                    # Ensure same length for comparison
                    min_len = min(len(feat_i), len(feat_k))
                    if min_len > 0:
                        ai, ak = _dtw_align_1d(feat_i, feat_k)
                        if np.all(ai == 0) and np.all(ak == 0):
                            sim = 1.0
                        else:
                            dot_product = np.dot(ai, ak)
                            norm_i = np.linalg.norm(ai)
                            norm_k = np.linalg.norm(ak)
                            sim = 0.0 if (norm_i == 0 or norm_k == 0) else dot_product / (norm_i * norm_k)
                            sim = max(0, min(1, sim))
                        
                        feature_similarities.append(sim)
            
            row[f'{feature_name}_similarity'] = np.mean(feature_similarities) if feature_similarities else 0.0
        
        row['overall_similarity'] = track_avg_similarities[i]
        track_data.append(row)
    
    # Create DataFrame and sort by overall similarity
    df = pd.DataFrame(track_data)
    df = df.sort_values('overall_similarity', ascending=False)
    
    # Reorder to put least similar pair at first and last positions
    least_sim_idx1, least_sim_idx2 = least_similar_pair
    least_sim_track1 = track_names[least_sim_idx1]
    least_sim_track2 = track_names[least_sim_idx2]
    
    # Find these tracks in the sorted dataframe
    track1_row = df[df['track_name'] == least_sim_track1].iloc[0]
    track2_row = df[df['track_name'] == least_sim_track2].iloc[0]
    
    # Remove them from the dataframe
    df_filtered = df[~df['track_name'].isin([least_sim_track1, least_sim_track2])]
    
    # Create new dataframe with least similar pair at first and last
    new_df = pd.DataFrame([track1_row])  # First row
    new_df = pd.concat([new_df, df_filtered], ignore_index=True)  # Middle rows
    new_df = pd.concat([new_df, pd.DataFrame([track2_row])], ignore_index=True)  # Last row
    
    csv_path = output_dir / 'track_similarities.csv'
    new_df.to_csv(csv_path, index=False)
    
    print(f"✓ Track similarity analysis saved: {csv_path}")
    print(f"  - {len(new_df)} tracks analyzed")
    print(f"  - Individual similarities for {len(key_features)} features")
    print(f"  - Overall similarity calculated as average with all other tracks")
    print(f"  - Least similar pair placed at first and last positions")
    
    # Print summary
    print(f"\n📊 Least similar pair (at first and last positions):")
    print(f"  {least_sim_track1[:30]}... ↔ {least_sim_track2[:30]}... : {min_similarity:.3f}")
    
    print(f"\n📊 Similarity range:")
    print(f"  Highest: {new_df['overall_similarity'].max():.3f}")
    print(f"  Lowest: {new_df['overall_similarity'].min():.3f}")
    print(f"  Average: {new_df['overall_similarity'].mean():.3f}")


def write_track_feature_summary(results: Dict[str, Any], output_dir: Path) -> None:
    """
    Write per-track feature summary (no pairwise) as CSV: one row per track with 5 features

    Columns: track_name, mfcc_mean, spectral_centroid_mean, spectral_rolloff_mean,
             spectral_bandwidth_mean, zcr_mean, rms_mean
    """
    track_names = results['track_names']
    track_results = results['track_results']

    key_features = ['mfcc', 'spectral_centroid', 'spectral_rolloff', 'spectral_bandwidth', 'zcr', 'rms']

    rows = []
    for track_name in track_names:
        track_result = track_results[track_name]
        row = {'track_name': track_name}

        for feature_name in key_features:
            feature_values: list = []
            for seg_features in track_result['raw_features']:
                if feature_name in seg_features:
                    feat = seg_features[feature_name]
                    if hasattr(feat, 'ndim') and feat.ndim > 1:
                        if feature_name == 'mfcc':
                            feature_values.extend(np.mean(feat, axis=1))
                        else:
                            feature_values.extend(np.mean(feat, axis=0))
                    else:
                        # scalar or 1D
                        try:
                            feature_values.extend(feat)
                        except TypeError:
                            feature_values.append(float(feat))

            mean_val = float(np.mean(feature_values)) if len(feature_values) > 0 else 0.0
            row[f'{feature_name}_mean'] = mean_val

        rows.append(row)

    df = pd.DataFrame(rows)
    csv_path = output_dir / 'track_feature_summary.csv'
    df.to_csv(csv_path, index=False)
    print(f"✓ Track feature summary saved: {csv_path}")

def plot_feature_distribution(results: Dict[str, Any], output_dir: Path) -> None:
    """
    Plot feature distribution comparison across tracks
    
    Args:
        results: Fine analysis results
        output_dir: Output directory
    """
    track_results = results['track_results']
    track_names = results['track_names']
    
    # Select 5 key features for distribution plot (tempo removed)
    key_features = ['mfcc', 'spectral_centroid', 'spectral_rolloff', 'spectral_bandwidth', 'zcr', 'rms']
    
    # Create individual plots for each track
    for track_name in track_names:
        track_result = track_results[track_name]
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()
        
        for i, feature_name in enumerate(key_features):
            ax = axes[i]
            
            # Collect feature values for this track
            feature_values = []
            for seg_features in track_result['raw_features']:
                if feature_name in seg_features:
                    feat = seg_features[feature_name]
                    if feat.ndim > 1:
                        # For multi-dimensional features, take mean across time
                        if feature_name == 'mfcc':
                            # For MFCC, take mean of all coefficients
                            feature_values.extend(np.mean(feat, axis=1))
                        else:
                            feature_values.extend(np.mean(feat, axis=0))
                    else:
                        feature_values.extend(feat)
            
            if feature_values:
                # Plot histogram
                ax.hist(feature_values, bins=20, alpha=0.7, 
                       color='skyblue', edgecolor='black', density=True)
                
                # Add statistics
                mean_val = np.mean(feature_values)
                std_val = np.std(feature_values)
                ax.axvline(mean_val, color='red', linestyle='--', linewidth=2, 
                          label=f'Mean: {mean_val:.3f}')
                ax.axvline(mean_val + std_val, color='orange', linestyle=':', alpha=0.7,
                          label=f'±1σ: {std_val:.3f}')
                ax.axvline(mean_val - std_val, color='orange', linestyle=':', alpha=0.7)
            
            ax.set_title(f'{feature_name.replace("_", " ").title()} Distribution', 
                        fontsize=12, fontweight='bold')
            ax.set_xlabel(f'{feature_name.replace("_", " ").title()}', fontsize=10)
            ax.set_ylabel('Density', fontsize=10)
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)
        
        # Clean track name for filename
        clean_name = "".join(c for c in track_name if c.isalnum() or c in (' ', '-', '_')).rstrip()
        clean_name = clean_name.replace(' ', '_')[:50]  # Limit filename length
        
        plt.suptitle(f'6-Feature Distribution: {track_name}', 
                     fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(output_dir / f'{clean_name}_feature_distributions.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Feature distribution plot saved: {output_dir / f'{clean_name}_feature_distributions.png'}")
    
    # Also create a combined comparison plot
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    colors = plt.cm.Set3(np.linspace(0, 1, len(track_names)))
    
    for i, feature_name in enumerate(key_features):
        ax = axes[i]
        
        for j, track_name in enumerate(track_names):
            track_result = track_results[track_name]
            
            # Collect feature values across all segments
            feature_values = []
            for seg_features in track_result['raw_features']:
                if feature_name in seg_features:
                    feat = seg_features[feature_name]
                    if feat.ndim > 1:
                        # For multi-dimensional features, take mean across time
                        if feature_name == 'mfcc':
                            # For MFCC, take mean of all coefficients
                            feature_values.extend(np.mean(feat, axis=1))
                        else:
                            feature_values.extend(np.mean(feat, axis=0))
                    else:
                        feature_values.extend(feat)
            
            if feature_values:
                # Plot histogram
                ax.hist(feature_values, bins=20, alpha=0.6, 
                       label=track_name[:15] + '...' if len(track_name) > 15 else track_name,
                       color=colors[j], density=True)
        
        ax.set_title(f'{feature_name.replace("_", " ").title()} Distribution', 
                    fontsize=12, fontweight='bold')
        ax.set_xlabel(f'{feature_name.replace("_", " ").title()}', fontsize=10)
        ax.set_ylabel('Density', fontsize=10)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    
    plt.suptitle('6-Feature Distribution Comparison Across All Tracks', 
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / 'all_tracks_feature_distributions.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Combined feature distribution plot saved: {output_dir / 'all_tracks_feature_distributions.png'}")


def plot_segment_analysis(results: Dict[str, Any], output_dir: Path) -> None:
    """
    Plot segment-level analysis
    
    Args:
        results: Fine analysis results
        output_dir: Output directory
    """
    track_results = results['track_results']
    track_names = results['track_names']
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # 1. Number of segments per track
    ax1 = axes[0, 0]
    segment_counts = [len(track_results[name]['segments']) for name in track_names]
    display_names = [name[:10] + '...' if len(name) > 10 else name for name in track_names]
    bars = ax1.bar(range(len(track_names)), segment_counts, color='skyblue', alpha=0.7)
    ax1.set_title('Number of Segments per Track', fontweight='bold')
    ax1.set_xlabel('Tracks')
    ax1.set_ylabel('Segment Count')
    ax1.set_xticks(range(len(track_names)))
    ax1.set_xticklabels(display_names, rotation=45, ha='right')
    
    # Add value labels on bars
    for bar, count in zip(bars, segment_counts):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, 
                str(count), ha='center', va='bottom')
    
    # 2. Average segment duration
    ax2 = axes[0, 1]
    avg_durations = []
    for name in track_names:
        track_result = track_results[name]
        segment_times = track_result['segment_times']
        if len(segment_times) > 1:
            durations = np.diff(segment_times)
            avg_durations.append(np.mean(durations))
        else:
            avg_durations.append(0.5)  # Default segment duration
    
    bars = ax2.bar(range(len(track_names)), avg_durations, color='lightcoral', alpha=0.7)
    ax2.set_title('Average Segment Duration', fontweight='bold')
    ax2.set_xlabel('Tracks')
    ax2.set_ylabel('Duration (seconds)')
    ax2.set_xticks(range(len(track_names)))
    ax2.set_xticklabels(display_names, rotation=45, ha='right')
    
    # Add value labels on bars
    for bar, duration in zip(bars, avg_durations):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                f'{duration:.2f}s', ha='center', va='bottom')
    
    # 3. Feature dimension
    ax3 = axes[1, 0]
    feature_dims = [track_results[name]['fused_features'].shape[1] for name in track_names]
    bars = ax3.bar(range(len(track_names)), feature_dims, color='lightgreen', alpha=0.7)
    ax3.set_title('Fused Feature Dimension', fontweight='bold')
    ax3.set_xlabel('Tracks')
    ax3.set_ylabel('Feature Dimension')
    ax3.set_xticks(range(len(track_names)))
    ax3.set_xticklabels(display_names, rotation=45, ha='right')
    
    # Add value labels on bars
    for bar, dim in zip(bars, feature_dims):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 10, 
                str(dim), ha='center', va='bottom')
    
    # 4. Total analysis time
    ax4 = axes[1, 1]
    total_times = [track_results[name]['segment_times'][-1] if len(track_results[name]['segment_times']) > 0 else 0 
                   for name in track_names]
    bars = ax4.bar(range(len(track_names)), total_times, color='gold', alpha=0.7)
    ax4.set_title('Total Track Duration', fontweight='bold')
    ax4.set_xlabel('Tracks')
    ax4.set_ylabel('Duration (seconds)')
    ax4.set_xticks(range(len(track_names)))
    ax4.set_xticklabels(display_names, rotation=45, ha='right')
    
    # Add value labels on bars
    for bar, time in zip(bars, total_times):
        ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
                f'{time:.1f}s', ha='center', va='bottom')
    
    plt.suptitle('Segment Analysis Summary', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / 'segment_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Segment analysis plot saved: {output_dir / 'segment_analysis.png'}")


def create_all_visualizations(results: Dict[str, Any], output_dir: Path, use_dtw: bool = False) -> None:
    """
    Create all visualization plots
    
    Args:
        results: Fine analysis results
        output_dir: Output directory
    """
    print("[timbre-sequence-analyze] Creating visualizations...")
    
    # Create visualizations directory
    viz_dir = output_dir / "visualizations"
    viz_dir.mkdir(exist_ok=True)
    
    # List of visualization functions to try
    viz_functions = [
        ("Feature Trends", plot_feature_trends),
        ("Hierarchical Clustering", plot_hierarchical_clustering),
        ("Feature Distributions", plot_feature_distribution),
        ("Segment Analysis", plot_segment_analysis),
        ("Detailed Similarities", lambda r, d: calculate_detailed_similarities(r, d, use_dtw=use_dtw)),
        ("Track Feature Summary", write_track_feature_summary),
        ("Track Relationship Scatter", plot_track_relationship_scatter)
    ]
    
    successful_viz = 0
    
    for viz_name, viz_func in viz_functions:
        try:
            print(f"[timbre-sequence-analyze] Creating {viz_name}...")
            viz_func(results, viz_dir)
            successful_viz += 1
        except Exception as e:
            print(f"[timbre-sequence-analyze] Error creating {viz_name}: {e}")
            import traceback
            traceback.print_exc()
    
    print(f"[timbre-sequence-analyze] Successfully created {successful_viz}/{len(viz_functions)} visualizations")
    print(f"[timbre-sequence-analyze] Visualizations saved to: {viz_dir}")


def plot_track_relationship_scatter(results: Dict[str, Any], output_dir: Path) -> None:
    """
    Create 2D scatter plot showing track relationships based on fused features
    Similar to the Beethoven-style visualization showing track proximity
    
    Args:
        results: Fine analysis results
        output_dir: Output directory
    """
    from sklearn.manifold import TSNE
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
    
    track_names = results['track_names']
    track_results = results['track_results']
    
    if len(track_names) < 2:
        print("⚠️ Need at least 2 tracks for relationship scatter plot")
        return
    
    # Extract fused features for each track
    track_features = []
    for track_name in track_names:
        track_result = track_results[track_name]
        # Use mean of fused features across all segments
        fused_features = track_result['fused_features']
        mean_features = np.mean(fused_features, axis=0)
        track_features.append(mean_features)
    
    track_features = np.array(track_features)
    
    # Standardize features
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(track_features)
    
    # Dimensionality reduction to 2D
    if len(track_names) <= 3:
        # For very few tracks, use PCA
        reducer = PCA(n_components=2, random_state=42)
        embedding = reducer.fit_transform(features_scaled)
    else:
        # Use t-SNE for better visualization of relationships
        try:
            reducer = TSNE(n_components=2, random_state=42, perplexity=min(30, len(track_names)-1))
            embedding = reducer.fit_transform(features_scaled)
        except:
            # Fallback to PCA if t-SNE fails
            reducer = PCA(n_components=2, random_state=42)
            embedding = reducer.fit_transform(features_scaled)
    
    # Create the scatter plot
    plt.figure(figsize=(12, 10))
    
    # Define colors for different tracks (cycle through a palette)
    colors = plt.cm.Set3(np.linspace(0, 1, len(track_names)))
    
    # Plot each track
    for i, (track_name, color) in enumerate(zip(track_names, colors)):
        # Truncate track name for display
        display_name = track_name.split()[0] if ' ' in track_name else track_name[:10]
        
        plt.scatter(embedding[i, 0], embedding[i, 1], 
                   c=[color], s=200, alpha=0.8, edgecolors='black', linewidth=2)
        
        # Add track name annotation
        plt.annotate(display_name, 
                    (embedding[i, 0], embedding[i, 1]),
                    xytext=(5, 5), textcoords='offset points',
                    fontsize=10, fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
    
    # Add grid
    plt.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    plt.grid(True, alpha=0.1, linestyle=':', linewidth=0.3, which='minor')
    plt.minorticks_on()
    
    # Labels and title
    plt.xlabel('Dimension 1', fontsize=12, fontweight='bold')
    plt.ylabel('Dimension 2', fontsize=12, fontweight='bold')
    plt.title('Track Relationship Visualization\n(Based on Fused Audio Features)', 
              fontsize=16, fontweight='bold', pad=20)
    
    # Add legend with track names and colors
    legend_elements = []
    for i, (track_name, color) in enumerate(zip(track_names, colors)):
        display_name = track_name.split()[0] if ' ' in track_name else track_name[:15]
        legend_elements.append(plt.Line2D([0], [0], marker='o', color='w', 
                                        markerfacecolor=color, markersize=10, 
                                        label=display_name))
    
    plt.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1.0, 1.0),
              fontsize=10, title='Tracks', title_fontsize=12)
    
    # Add distance information
    if len(track_names) >= 2:
        # Calculate pairwise distances
        from scipy.spatial.distance import pdist, squareform
        distances = pdist(embedding, metric='euclidean')
        distance_matrix = squareform(distances)
        
        # Find closest and farthest pairs
        min_dist_idx = np.unravel_index(np.argmin(distance_matrix + np.eye(len(track_names)) * np.inf), 
                                       distance_matrix.shape)
        max_dist_idx = np.unravel_index(np.argmax(distance_matrix), distance_matrix.shape)
        
        min_dist = distance_matrix[min_dist_idx]
        max_dist = distance_matrix[max_dist_idx]
        
        # Add text box with distance info
        info_text = f'Closest pair: {min_dist:.2f}\nFarthest pair: {max_dist:.2f}'
        plt.text(0.02, 0.98, info_text, transform=plt.gca().transAxes, 
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
                fontsize=10)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'track_relationship_scatter.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Track relationship scatter plot saved: {output_dir / 'track_relationship_scatter.png'}")
