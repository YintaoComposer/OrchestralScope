from typing import List, Tuple

import numpy as np
from sklearn.metrics import silhouette_score
from sklearn.mixture import GaussianMixture


def distribution_vector(labels: np.ndarray, num_classes: int) -> np.ndarray:
    counts = np.bincount(labels, minlength=num_classes).astype(float)
    p = counts / max(1, counts.sum())
    return p


def markov_matrix(labels: np.ndarray, num_classes: int, eps: float = 1e-8) -> np.ndarray:
    T = np.zeros((num_classes, num_classes), dtype=float)
    for a, b in zip(labels[:-1], labels[1:]):
        T[a, b] += 1
    row_sum = T.sum(axis=1, keepdims=True) + eps
    return T / row_sum


def js_divergence(p: np.ndarray, q: np.ndarray, eps: float = 1e-12) -> float:
    p = p + eps
    q = q + eps
    m = 0.5 * (p + q)
    def kl(a, b):
        return float(np.sum(a * (np.log(a) - np.log(b))))
    return 0.5 * kl(p, m) + 0.5 * kl(q, m)


def levenshtein(a: List[int], b: List[int]) -> int:
    n, m = len(a), len(b)
    dp = np.zeros((n + 1, m + 1), dtype=int)
    for i in range(n + 1):
        dp[i, 0] = i
    for j in range(m + 1):
        dp[0, j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            dp[i, j] = min(dp[i - 1, j] + 1, dp[i, j - 1] + 1, dp[i - 1, j - 1] + cost)
    return int(dp[n, m])


def frobenius_distance(A: np.ndarray, B: np.ndarray) -> float:
    return float(np.linalg.norm(A - B))


def combined_distance(d_order: float, d_dist: float, alpha: float = 0.6) -> float:
    return alpha * d_order + (1.0 - alpha) * d_dist


def cosine_silhouette_score(X: np.ndarray, labels: np.ndarray) -> float:
    """
    Calculate cosine distance-based Silhouette score
    
    Args:
        X: Feature matrix (n_samples, n_features)
        labels: Cluster labels (n_samples,)
    
    Returns:
        Cosine Silhouette score
    """
    if len(set(labels)) < 2:
        return -1.0
    
    # Calculate cosine similarity matrix
    X_norm = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-8)
    cosine_sim = np.dot(X_norm, X_norm.T)
    cosine_dist = 1 - cosine_sim
    
    n_samples = len(labels)
    silhouette_scores = np.zeros(n_samples)
    
    for i in range(n_samples):
        # Calculate intra-cluster average distance (a_i)
        cluster_i = labels[i]
        same_cluster_mask = labels == cluster_i
        same_cluster_mask[i] = False  # Exclude self
        if same_cluster_mask.sum() == 0:
            a_i = 0
        else:
            a_i = cosine_dist[i, same_cluster_mask].mean()
        
        # Calculate nearest neighbor cluster average distance (b_i)
        other_clusters = [c for c in set(labels) if c != cluster_i]
        if not other_clusters:
            b_i = 0
        else:
            min_avg_dist = float('inf')
            for cluster_j in other_clusters:
                other_cluster_mask = labels == cluster_j
                if other_cluster_mask.sum() > 0:
                    avg_dist = cosine_dist[i, other_cluster_mask].mean()
                    min_avg_dist = min(min_avg_dist, avg_dist)
            b_i = min_avg_dist if min_avg_dist != float('inf') else 0
        
        # Calculate Silhouette score
        if max(a_i, b_i) == 0:
            silhouette_scores[i] = 0
        else:
            silhouette_scores[i] = (b_i - a_i) / max(a_i, b_i)
    
    return float(silhouette_scores.mean())


def gap_statistic(X: np.ndarray, labels: np.ndarray, n_bootstrap: int = 10, random_state: int = 42) -> float:
    """
    Calculate Gap statistic
    
    Args:
        X: Feature matrix (n_samples, n_features)
        labels: Cluster labels (n_samples,)
        n_bootstrap: Number of bootstrap samples
        random_state: Random seed
    
    Returns:
        Gap statistic value
    """
    if len(set(labels)) < 2:
        return 0.0
    
    np.random.seed(random_state)
    n_samples, n_features = X.shape
    
    # Calculate within-cluster sum of squares for original data
    def within_cluster_sum_of_squares(X, labels):
        wcss = 0.0
        for cluster_id in set(labels):
            cluster_mask = labels == cluster_id
            if cluster_mask.sum() > 1:
                cluster_center = X[cluster_mask].mean(axis=0)
                wcss += np.sum((X[cluster_mask] - cluster_center) ** 2)
        return wcss
    
    log_wcss_original = np.log(within_cluster_sum_of_squares(X, labels) + 1e-8)
    
    # Generate uniformly distributed bootstrap samples and calculate expected log(W_k)
    log_wcss_bootstrap = []
    for _ in range(n_bootstrap):
        # Generate uniformly distributed samples within the bounding box of original data
        X_min = X.min(axis=0)
        X_max = X.max(axis=0)
        X_bootstrap = np.random.uniform(X_min, X_max, size=(n_samples, n_features))
        
        # Perform same clustering on bootstrap samples
        try:
            gmm = GaussianMixture(
                n_components=len(set(labels)),
                covariance_type='full',
                random_state=random_state,
                n_init=1
            )
            labels_bootstrap = gmm.fit_predict(X_bootstrap)
            wcss_bootstrap = within_cluster_sum_of_squares(X_bootstrap, labels_bootstrap)
            log_wcss_bootstrap.append(np.log(wcss_bootstrap + 1e-8))
        except:
            continue
    
    if not log_wcss_bootstrap:
        return 0.0
    
    expected_log_wcss = np.mean(log_wcss_bootstrap)
    gap = expected_log_wcss - log_wcss_original
    
    return float(gap)


def weighted_feature_silhouette_score(X: np.ndarray, labels: np.ndarray, feature_weights: dict = None) -> float:
    """
    Calculate feature weight-based Silhouette score
    
    Args:
        X: Feature matrix (n_samples, n_features)
        labels: Cluster labels (n_samples,)
        feature_weights: Feature weight dictionary, format like {'mfcc': 0.5, 'spectral_centroid': 0.2, ...}
    
    Returns:
        Weighted Silhouette score
    """
    if len(set(labels)) < 2:
        return -1.0
    
    # Default feature weights (based on timbre-fine-analyze weights)
    if feature_weights is None:
        feature_weights = {
            'mfcc': 0.50,           # 50% - Main feature, multi-dimensional contribution, maximum information gain
            'spectral_centroid': 0.20,  # 20% - Timbre brightness, medium-high information gain
            'spectral_bandwidth': 0.10, # 10% - Timbre complexity, complementary to Centroid
            'zcr': 0.10,            # 10% - Helpful in speech/percussion identification, increased weight
            'spectral_rolloff': 0.06,   # 6% - Moderate contribution in some models, reduced weight
            'rms': 0.04             # 4% - Represents energy, complementary to other features, but weak alone, reduced weight
        }
    
    # Assume feature order: MFCC(20) + Delta(20) + Delta2(20) + Chroma(12) + SpectralFlux(1)
    # This needs to be adjusted according to actual feature extraction order
    n_mfcc = 20
    n_chroma = 12
    n_flux = 1
    
    # Create feature weight vector
    feature_weight_vector = np.ones(X.shape[1])
    
    # MFCC part (first 20 dimensions)
    if X.shape[1] >= n_mfcc:
        feature_weight_vector[:n_mfcc] *= feature_weights.get('mfcc', 0.5)
    
    # Delta MFCC part (dimensions 21-40)
    if X.shape[1] >= 2 * n_mfcc:
        feature_weight_vector[n_mfcc:2*n_mfcc] *= feature_weights.get('mfcc', 0.5) * 0.8  # Delta weight slightly lower
    
    # Delta2 MFCC part (dimensions 41-60)
    if X.shape[1] >= 3 * n_mfcc:
        feature_weight_vector[2*n_mfcc:3*n_mfcc] *= feature_weights.get('mfcc', 0.5) * 0.6  # Delta2 weight even lower
    
    # Chroma part (dimensions 61-72)
    if X.shape[1] >= 3 * n_mfcc + n_chroma:
        start_idx = 3 * n_mfcc
        end_idx = start_idx + n_chroma
        feature_weight_vector[start_idx:end_idx] *= feature_weights.get('chroma', 0.1)
    
    # Spectral Flux part (last dimension)
    if X.shape[1] > 0:
        feature_weight_vector[-1] *= feature_weights.get('scflux', 0.04)
    
    # Normalize weight vector
    feature_weight_vector = feature_weight_vector / np.sum(feature_weight_vector) * X.shape[1]
    
    # Calculate weighted distance matrix
    n_samples = len(labels)
    silhouette_scores = np.zeros(n_samples)
    
    for i in range(n_samples):
        # Calculate intra-cluster average distance (a_i)
        cluster_i = labels[i]
        same_cluster_mask = labels == cluster_i
        same_cluster_mask[i] = False  # Exclude self
        
        if same_cluster_mask.sum() == 0:
            a_i = 0
        else:
            # Weighted Euclidean distance
            diff = X[i] - X[same_cluster_mask]
            weighted_dist = np.sqrt(np.sum((diff ** 2) * feature_weight_vector, axis=1))
            a_i = weighted_dist.mean()
        
        # Calculate nearest neighbor cluster average distance (b_i)
        other_clusters = [c for c in set(labels) if c != cluster_i]
        if not other_clusters:
            b_i = 0
        else:
            min_avg_dist = float('inf')
            for cluster_j in other_clusters:
                other_cluster_mask = labels == cluster_j
                if other_cluster_mask.sum() > 0:
                    diff = X[i] - X[other_cluster_mask]
                    weighted_dist = np.sqrt(np.sum((diff ** 2) * feature_weight_vector, axis=1))
                    avg_dist = weighted_dist.mean()
                    min_avg_dist = min(min_avg_dist, avg_dist)
            b_i = min_avg_dist if min_avg_dist != float('inf') else 0
        
        # Calculate Silhouette score
        if max(a_i, b_i) == 0:
            silhouette_scores[i] = 0
        else:
            silhouette_scores[i] = (b_i - a_i) / max(a_i, b_i)
    
    return float(silhouette_scores.mean())


def feature_weighted_gap_statistic(X: np.ndarray, labels: np.ndarray, feature_weights: dict = None, n_bootstrap: int = 10, random_state: int = 42) -> float:
    """
    Feature weight-based Gap statistic
    
    Args:
        X: Feature matrix (n_samples, n_features)
        labels: Cluster labels (n_samples,)
        feature_weights: Feature weight dictionary
        n_bootstrap: Number of bootstrap samples
        random_state: Random seed
    
    Returns:
        Weighted Gap statistic value
    """
    if len(set(labels)) < 2:
        return 0.0
    
    # Default feature weights (based on timbre-fine-analyze weights)
    if feature_weights is None:
        feature_weights = {
            'mfcc': 0.50,           # 50% - Main feature, multi-dimensional contribution, maximum information gain
            'spectral_centroid': 0.20,  # 20% - Timbre brightness, medium-high information gain
            'spectral_bandwidth': 0.10, # 10% - Timbre complexity, complementary to Centroid
            'zcr': 0.10,            # 10% - Helpful in speech/percussion identification, increased weight
            'spectral_rolloff': 0.06,   # 6% - Moderate contribution in some models, reduced weight
            'rms': 0.04             # 4% - Represents energy, complementary to other features, but weak alone, reduced weight
        }
    
    # Create feature weight vector (same logic as above)
    feature_weight_vector = np.ones(X.shape[1])
    n_mfcc = 20
    n_chroma = 12
    
    if X.shape[1] >= n_mfcc:
        feature_weight_vector[:n_mfcc] *= feature_weights.get('mfcc', 0.5)
    if X.shape[1] >= 2 * n_mfcc:
        feature_weight_vector[n_mfcc:2*n_mfcc] *= feature_weights.get('mfcc', 0.5) * 0.8
    if X.shape[1] >= 3 * n_mfcc:
        feature_weight_vector[2*n_mfcc:3*n_mfcc] *= feature_weights.get('mfcc', 0.5) * 0.6
    if X.shape[1] >= 3 * n_mfcc + n_chroma:
        start_idx = 3 * n_mfcc
        end_idx = start_idx + n_chroma
        feature_weight_vector[start_idx:end_idx] *= feature_weights.get('chroma', 0.1)
    if X.shape[1] > 0:
        feature_weight_vector[-1] *= feature_weights.get('scflux', 0.04)
    
    feature_weight_vector = feature_weight_vector / np.sum(feature_weight_vector) * X.shape[1]
    
    np.random.seed(random_state)
    n_samples, n_features = X.shape
    
    # Calculate weighted within-cluster sum of squares
    def weighted_within_cluster_sum_of_squares(X, labels, weights):
        wcss = 0.0
        for cluster_id in set(labels):
            cluster_mask = labels == cluster_id
            if cluster_mask.sum() > 1:
                cluster_center = X[cluster_mask].mean(axis=0)
                diff = X[cluster_mask] - cluster_center
                weighted_diff = (diff ** 2) * weights
                wcss += np.sum(weighted_diff)
        return wcss
    
    log_wcss_original = np.log(weighted_within_cluster_sum_of_squares(X, labels, feature_weight_vector) + 1e-8)
    
    # Generate weighted bootstrap samples
    log_wcss_bootstrap = []
    for _ in range(n_bootstrap):
        X_min = X.min(axis=0)
        X_max = X.max(axis=0)
        X_bootstrap = np.random.uniform(X_min, X_max, size=(n_samples, n_features))
        
        try:
            from sklearn.mixture import GaussianMixture
            gmm = GaussianMixture(
                n_components=len(set(labels)),
                covariance_type='full',
                random_state=random_state,
                n_init=1
            )
            labels_bootstrap = gmm.fit_predict(X_bootstrap)
            wcss_bootstrap = weighted_within_cluster_sum_of_squares(X_bootstrap, labels_bootstrap, feature_weight_vector)
            log_wcss_bootstrap.append(np.log(wcss_bootstrap + 1e-8))
        except:
            continue
    
    if not log_wcss_bootstrap:
        return 0.0
    
    expected_log_wcss = np.mean(log_wcss_bootstrap)
    gap = expected_log_wcss - log_wcss_original
    
    return float(gap)


def conservative_silhouette_score(X: np.ndarray, labels: np.ndarray, feature_weights: dict = None, 
                                 min_cluster_size_ratio: float = 0.05) -> float:
    """
    Conservative Silhouette score calculation to reduce over-segmentation
    
    Args:
        X: Feature matrix (n_samples, n_features)
        labels: Cluster labels (n_samples,)
        feature_weights: Feature weight dictionary
        min_cluster_size_ratio: Minimum cluster size ratio
    
    Returns:
        Conservative Silhouette score
    """
    if len(set(labels)) < 2:
        return -1.0
    
    # Check cluster sizes, filter out clusters that are too small
    n_samples = len(labels)
    min_cluster_size = max(1, int(n_samples * min_cluster_size_ratio))
    
    valid_clusters = []
    for cluster_id in set(labels):
        cluster_size = np.sum(labels == cluster_id)
        if cluster_size >= min_cluster_size:
            valid_clusters.append(cluster_id)
    
    if len(valid_clusters) < 2:
        return -1.0
    
    # Only calculate Silhouette score for valid clusters
    valid_mask = np.isin(labels, valid_clusters)
    if np.sum(valid_mask) < 2:
        return -1.0
    
    X_valid = X[valid_mask]
    labels_valid = labels[valid_mask]
    
    # Use weighted Silhouette score: only use MFCC
    if feature_weights is None:
        feature_weights = {
            'mfcc': 1.00,
            'spectral_centroid': 0.00,
            'spectral_bandwidth': 0.00,
            'zcr': 0.00,
            'spectral_rolloff': 0.00,
            'rms': 0.00
        }
    
    # Create feature weight vector
    feature_weight_vector = np.ones(X_valid.shape[1])
    n_mfcc = 20
    n_chroma = 12
    
    if X_valid.shape[1] >= n_mfcc:
        feature_weight_vector[:n_mfcc] *= feature_weights.get('mfcc', 0.5)
    if X_valid.shape[1] >= 2 * n_mfcc:
        feature_weight_vector[n_mfcc:2*n_mfcc] *= feature_weights.get('mfcc', 0.5) * 0.8
    if X_valid.shape[1] >= 3 * n_mfcc:
        feature_weight_vector[2*n_mfcc:3*n_mfcc] *= feature_weights.get('mfcc', 0.5) * 0.6
    if X_valid.shape[1] >= 3 * n_mfcc + n_chroma:
        start_idx = 3 * n_mfcc
        end_idx = start_idx + n_chroma
        feature_weight_vector[start_idx:end_idx] *= feature_weights.get('chroma', 0.1)
    if X_valid.shape[1] > 0:
        feature_weight_vector[-1] *= feature_weights.get('scflux', 0.04)
    
    feature_weight_vector = feature_weight_vector / np.sum(feature_weight_vector) * X_valid.shape[1]
    
    # Calculate conservative Silhouette score
    silhouette_scores = np.zeros(len(labels_valid))
    
    for i in range(len(labels_valid)):
        cluster_i = labels_valid[i]
        same_cluster_mask = labels_valid == cluster_i
        same_cluster_mask[i] = False
        
        if same_cluster_mask.sum() == 0:
            a_i = 0
        else:
            diff = X_valid[i] - X_valid[same_cluster_mask]
            weighted_dist = np.sqrt(np.sum((diff ** 2) * feature_weight_vector, axis=1))
            a_i = weighted_dist.mean()
        
        other_clusters = [c for c in valid_clusters if c != cluster_i]
        if not other_clusters:
            b_i = 0
        else:
            min_avg_dist = float('inf')
            for cluster_j in other_clusters:
                other_cluster_mask = labels_valid == cluster_j
                if other_cluster_mask.sum() > 0:
                    diff = X_valid[i] - X_valid[other_cluster_mask]
                    weighted_dist = np.sqrt(np.sum((diff ** 2) * feature_weight_vector, axis=1))
                    avg_dist = weighted_dist.mean()
                    min_avg_dist = min(min_avg_dist, avg_dist)
            b_i = min_avg_dist if min_avg_dist != float('inf') else 0
        
        if max(a_i, b_i) == 0:
            silhouette_scores[i] = 0
        else:
            silhouette_scores[i] = (b_i - a_i) / max(a_i, b_i)
    
    return float(silhouette_scores.mean())


def get_conservative_feature_weights(sensitivity: str = "normal") -> dict:
    """
    Get conservative feature weights based on sensitivity
    
    Args:
        sensitivity: Sensitivity level (conservative|normal|aggressive)
    
    Returns:
        Feature weight dictionary
    """
    if sensitivity == "conservative":
        # Conservative mode: based on timbre-fine-analyze weights, but emphasize main features more
        return {
            'mfcc': 0.60,           # Increase MFCC weight
            'spectral_centroid': 0.25,  # Increase spectral centroid weight
            'spectral_bandwidth': 0.08, # Decrease bandwidth weight
            'zcr': 0.05,            # Significantly reduce ZCR weight
            'spectral_rolloff': 0.02,   # Significantly reduce rolloff weight
            'rms': 0.00             # Completely ignore RMS
        }
    elif sensitivity == "aggressive":
        # Aggressive mode: based on timbre-fine-analyze weights, but emphasize detail features more
        return {
            'mfcc': 0.40,
            'spectral_centroid': 0.15,
            'spectral_bandwidth': 0.15,
            'zcr': 0.15,
            'spectral_rolloff': 0.10,
            'rms': 0.05
        }
    else:  # normal
        # Normal mode: use original timbre-fine-analyze weights
        return {
            'mfcc': 0.50,           # 50% - Main feature, multi-dimensional contribution, maximum information gain
            'spectral_centroid': 0.20,  # 20% - Timbre brightness, medium-high information gain
            'spectral_bandwidth': 0.10, # 10% - Timbre complexity, complementary to Centroid
            'zcr': 0.10,            # 10% - Helpful in speech/percussion identification, increased weight
            'spectral_rolloff': 0.06,   # 6% - Moderate contribution in some models, reduced weight
            'rms': 0.04             # 4% - Represents energy, complementary to other features, but weak alone, reduced weight
        }