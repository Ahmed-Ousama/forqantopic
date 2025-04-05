import numpy as np
from sklearn.cluster import (
    KMeans,
    DBSCAN,
    AgglomerativeClustering,
    SpectralClustering,
    Birch
)
from sklearn.mixture import GaussianMixture
import hdbscan

from sklearn.metrics import silhouette_score

SUPPORTED_CLUSTERING = [
    "kmeans",
    "hdbscan",
    "dbscan",
    "gmm",
    "agglomerative",
    "spectral",
    "birch"
]


def list_supported_clustering() -> list[str]:
    """
    Lists all available clustering algorithms.
    """
    return SUPPORTED_CLUSTERING


def estimate_optimal_k(embeddings, max_k: int = 10) -> int:
    """
    Estimate best number of clusters using silhouette score for KMeans/GMM.
    """
    best_score = -1
    best_k = 2

    for k in range(2, max_k + 1):
        kmeans = KMeans(n_clusters=k, random_state=42)
        labels = kmeans.fit_predict(embeddings)
        score = silhouette_score(embeddings, labels)
        if score > best_score:
            best_score = score
            best_k = k

    return best_k


def cluster_embeddings(
    reduced_embeddings: np.ndarray,
    method: str = "hdbscan",
    n_clusters: int | None = None,
    max_k: int = 10,
) -> np.ndarray | tuple[np.ndarray, object]:
    """
    Cluster reduced embeddings using the specified method.

    Parameters:
        reduced_embeddings: 2D array after dimensionality reduction
        method: Clustering algorithm name
        n_clusters: Optional (used by kmeans, gmm, spectral, agglomerative)
        max_k: Used for automatic cluster count estimation

    Returns:
        labels or (labels, model)
    """
    method = method.lower()
    X = np.array(reduced_embeddings)
    
    print(f"Clustering using {method} with n_clusters={n_clusters}")

    if method == "kmeans":
        if n_clusters is None:
            n_clusters = estimate_optimal_k(X, max_k)
            print(f"Auto-selected n_clusters={n_clusters} using silhouette score")
        model = KMeans(n_clusters=n_clusters, random_state=42)
        labels = model.fit_predict(X)

    elif method == "gmm":
        if n_clusters is None:
            n_clusters = estimate_optimal_k(X, max_k)
        model = GaussianMixture(n_components=n_clusters, random_state=42)
        labels = model.fit(X).predict(X)

    elif method == "hdbscan":
        model = hdbscan.HDBSCAN(min_cluster_size=3, prediction_data=True, min_samples=2)
        labels = model.fit_predict(X)

    elif method == "dbscan":
        model = DBSCAN(eps=0.5, min_samples=5, metric="euclidean")
        labels = model.fit_predict(X)

    elif method == "agglomerative":
        if n_clusters is None:
            raise ValueError("n_clusters must be provided for AgglomerativeClustering.")
        model = AgglomerativeClustering(n_clusters=n_clusters)
        labels = model.fit_predict(X)

    elif method == "spectral":
        if n_clusters is None:
            raise ValueError("n_clusters must be provided for SpectralClustering.")
        model = SpectralClustering(n_clusters=n_clusters, random_state=42, affinity='nearest_neighbors')
        labels = model.fit_predict(X)

    elif method == "birch":
        if n_clusters is None:
            raise ValueError("n_clusters must be provided for Birch.")
        model = Birch(n_clusters=n_clusters)
        labels = model.fit_predict(X)

    else:
        raise ValueError(f"Unknown clustering method: '{method}'. Supported: {SUPPORTED_CLUSTERING}")

    return (labels, model)
