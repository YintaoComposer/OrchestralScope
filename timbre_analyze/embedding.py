from typing import Tuple

import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap


def embed_features(X: np.ndarray, method: str = "umap", dim: int = 2, random_state: int = 42) -> np.ndarray:
    method = method.lower()
    
    # Check data validity
    if X.shape[0] < 2:
        # Too few data points, return simple coordinates
        if X.shape[0] == 1:
            return np.zeros((1, dim))
        else:
            return np.zeros((0, dim))
    
    # Check if there are all-zero rows
    if np.all(X == 0):
        # All-zero data, return random coordinates
        return np.random.RandomState(random_state).randn(X.shape[0], dim) * 0.1
    
    if method == "pca":
        return PCA(n_components=dim, random_state=random_state).fit_transform(X)
    if method == "tsne":
        return TSNE(n_components=dim, random_state=random_state, init="pca").fit_transform(X)
    
    # default umap - add more parameters to avoid errors
    try:
        reducer = umap.UMAP(
            n_components=dim, 
            random_state=random_state,
            n_neighbors=min(15, X.shape[0] - 1),  # Avoid neighbors exceeding number of data points
            min_dist=0.1,
            spread=1.0
        )
        return reducer.fit_transform(X)
    except Exception as e:
        print(f"[timbre-analyze] UMAP failed ({e}), falling back to PCA")
        return PCA(n_components=dim, random_state=random_state).fit_transform(X)


