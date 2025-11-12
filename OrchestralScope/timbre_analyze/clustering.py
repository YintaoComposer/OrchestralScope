from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, davies_bouldin_score
from .metrics import cosine_silhouette_score, gap_statistic, weighted_feature_silhouette_score, feature_weighted_gap_statistic, conservative_silhouette_score, get_conservative_feature_weights


@dataclass
class KSelectConfig:
    method: str = "bic"  # bic|aic|silhouette|db|combined|feature_weighted|conservative
    k_min: int = 4
    k_max: int = 12
    epsilon_ratio: float = 0.01
    # Combined scoring weights
    bic_weight: float = 0.4
    cosine_silhouette_weight: float = 0.3
    gap_weight: float = 0.3
    # Feature weight K selection
    use_feature_weights: bool = False
    feature_weights: dict = None
    # Conservative clustering parameters
    clustering_sensitivity: str = "normal"
    min_cluster_size_ratio: float = 0.05


def fit_gmm_k_candidates(
    X: np.ndarray,
    k_min: int,
    k_max: int,
    covariance_type: str = "full",
    n_init: int = 3,
    max_iter: int = 300,
    reg_covar: float = 1e-6,
    random_state: int = 42,
) -> Dict[int, GaussianMixture]:
    models: Dict[int, GaussianMixture] = {}
    n_samples = X.shape[0]
    
    # Adjust k_max to not exceed number of samples
    k_max = min(k_max, n_samples)
    k_min = min(k_min, n_samples)
    
    for k in range(k_min, k_max + 1):
        if k <= n_samples:  # Additional safety check
            try:
                gmm = GaussianMixture(
                    n_components=k,
                    covariance_type=covariance_type,
                    n_init=n_init,
                    max_iter=max_iter,
                    reg_covar=reg_covar,
                    random_state=random_state,
                )
                gmm.fit(X)
                models[k] = gmm
            except Exception:
                # Fallback: if fitting fails, skip this K
                continue
    return models


def select_k(X: np.ndarray, models: Dict[int, GaussianMixture], cfg: KSelectConfig) -> int:
    scores = {}
    if cfg.method in ("bic", "aic", "icl"):
        for k, m in models.items():
            if cfg.method == "bic":
                scores[k] = m.bic(X)
            elif cfg.method == "aic":
                scores[k] = m.aic(X)
            else:  # icl = bic + entropy penalty
                try:
                    bic = m.bic(X)
                    proba = m.predict_proba(X)
                    # Soft assignment entropy: -sum p*log p; add eps for numerical stability
                    eps = 1e-12
                    entropy = -np.sum(proba * np.log(proba + eps))
                    scores[k] = bic + entropy
                except Exception:
                    scores[k] = m.bic(X)
        # Prefer minimum score, but favor larger K when scores are close
        min_score = min(scores.values()) if scores else float("inf")
        # Allowed proximity threshold (relative difference)
        eps = getattr(cfg, "epsilon_ratio", 0.01)
        candidate_ks = [k for k, v in scores.items() if abs(v - min_score) <= eps * (abs(min_score) + 1e-9)]
        best_k = max(candidate_ks) if candidate_ks else min(scores, key=scores.get)
        # When neighboring K differences are below threshold, use secondary criterion
        sorted_items = sorted(scores.items())
        def rel_diff(a, b):
            return abs(a - b) / (abs(a) + 1e-9)
        neighbors = [k for k in (best_k - 1, best_k + 1) if k in scores]
        for nk in neighbors:
            if rel_diff(scores[best_k], scores[nk]) < cfg.epsilon_ratio:
                # Auxiliary metric: large silhouette/small DB
                labels = models[nk].predict(X)
                sil_nk = silhouette_score(X, labels) if len(set(labels)) > 1 else -1.0
                labels_best = models[best_k].predict(X)
                sil_best = silhouette_score(X, labels_best) if len(set(labels_best)) > 1 else -1.0
                # If silhouette is comparable or better, choose larger K; otherwise keep
                if sil_nk >= sil_best and nk > best_k:
                    best_k = nk
        return best_k
    elif cfg.method == "silhouette":
        for k, m in models.items():
            labels = m.predict(X)
            if len(set(labels)) < 2:
                scores[k] = -1.0
            else:
                scores[k] = silhouette_score(X, labels)
        return max(scores, key=scores.get)
    elif cfg.method == "combined":
        # Combined scoring: BIC + Cosine Silhouette + Gap statistic
        bic_scores = {}
        cosine_sil_scores = {}
        gap_scores = {}
        
        # Calculate scores for each metric
        for k, m in models.items():
            labels = m.predict(X)
            if len(set(labels)) < 2:
                bic_scores[k] = float('inf')
                cosine_sil_scores[k] = -1.0
                gap_scores[k] = 0.0
            else:
                # BIC score (lower is better)
                bic_scores[k] = m.bic(X)
                # Cosine Silhouette score (higher is better)
                cosine_sil_scores[k] = cosine_silhouette_score(X, labels)
                # Gap statistic (higher is better)
                gap_scores[k] = gap_statistic(X, labels)
        
        # Normalize each metric to [0,1] range
        def normalize_scores(scores_dict, reverse=False):
            if not scores_dict:
                return {}
            values = list(scores_dict.values())
            min_val, max_val = min(values), max(values)
            if max_val == min_val:
                return {k: 0.5 for k in scores_dict.keys()}
            normalized = {}
            for k, v in scores_dict.items():
                if reverse:  # For metrics where lower is better (e.g., BIC)
                    normalized[k] = (max_val - v) / (max_val - min_val)
                else:  # For metrics where higher is better
                    normalized[k] = (v - min_val) / (max_val - min_val)
            return normalized
        
        bic_norm = normalize_scores(bic_scores, reverse=True)
        cosine_sil_norm = normalize_scores(cosine_sil_scores, reverse=False)
        gap_norm = normalize_scores(gap_scores, reverse=False)
        
        # Combined scoring
        for k in models.keys():
            combined_score = (
                cfg.bic_weight * bic_norm.get(k, 0) +
                cfg.cosine_silhouette_weight * cosine_sil_norm.get(k, 0) +
                cfg.gap_weight * gap_norm.get(k, 0)
            )
            scores[k] = combined_score
        
        return max(scores, key=scores.get)
    elif cfg.method == "feature_weighted":
        # Feature weight-based K selection: only use weighted Silhouette score
        # Set default feature weights
        if cfg.feature_weights is None:
            feature_weights = {
                'mfcc': 0.50,
                'spectral_centroid': 0.20,
                'spectral_bandwidth': 0.10,
                'zcr': 0.10,
                'spectral_rolloff': 0.06,
                'rms': 0.04
            }
        else:
            feature_weights = cfg.feature_weights
        
        # Only calculate weighted Silhouette score
        for k, m in models.items():
            labels = m.predict(X)
            if len(set(labels)) < 2:
                scores[k] = -1.0
            else:
                # Only use weighted Silhouette score (higher is better)
                scores[k] = weighted_feature_silhouette_score(X, labels, feature_weights)
        
        return max(scores, key=scores.get)
    elif cfg.method == "conservative":
        # Conservative K selection: use conservative Silhouette score to reduce over-segmentation
        # Get feature weights based on sensitivity
        if cfg.feature_weights is None:
            feature_weights = get_conservative_feature_weights(cfg.clustering_sensitivity)
        else:
            feature_weights = cfg.feature_weights
        
        # Only calculate conservative Silhouette score
        for k, m in models.items():
            labels = m.predict(X)
            if len(set(labels)) < 2:
                scores[k] = -1.0
            else:
                # Use conservative Silhouette score (higher is better)
                scores[k] = conservative_silhouette_score(
                    X, labels, feature_weights, cfg.min_cluster_size_ratio
                )
        
        return max(scores, key=scores.get)
    else:  # db
        for k, m in models.items():
            labels = m.predict(X)
            if len(set(labels)) < 2:
                scores[k] = np.inf
            else:
                scores[k] = davies_bouldin_score(X, labels)
        return min(scores, key=scores.get)


def assign_global_labels(X: np.ndarray, gmm: GaussianMixture) -> np.ndarray:
    return gmm.predict(X)


def fit_gmm_auto_for_track(
    X: np.ndarray,
    k_min: int,
    k_max: int,
    select_method: str = "bic",
    epsilon_ratio: float = 0.01,
    covariance_type: str = "full",
    n_init: int = 3,
    max_iter: int = 300,
    random_state: int = 42,
) -> tuple[GaussianMixture, int]:
    models = fit_gmm_k_candidates(
        X,
        k_min,
        k_max,
        covariance_type=covariance_type,
        n_init=n_init,
        max_iter=max_iter,
        random_state=random_state,
    )
    k_sel = select_k(
        X,
        models,
        KSelectConfig(method=select_method, k_min=k_min, k_max=k_max, epsilon_ratio=epsilon_ratio),
    )
    return models[k_sel], k_sel


