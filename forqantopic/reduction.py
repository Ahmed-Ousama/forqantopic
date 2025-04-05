import numpy as np
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.manifold import TSNE
import umap

SUPPORTED_REDUCERS = ["pca", "umap", "tsne", "svd"]


def list_supported_reducers() -> list[str]:
    """
    Returns a list of all supported dimensionality reduction methods.
    """
    return SUPPORTED_REDUCERS


def hybrid_n_components(
    embeddings: np.ndarray | list,
    variance_threshold: float = 0.95,
    max_cap: int = 100,
    min_dim: int = 2
) -> int:
    embeddings = np.array(embeddings)
    n_samples, n_features = embeddings.shape

    # Safe cap
    max_valid = min(max_cap, n_samples, n_features)
    if max_valid < min_dim:
        max_valid = min_dim

    # Run PCA
    pca = PCA(n_components=max_valid).fit(embeddings)
    variances = pca.explained_variance_ratio_
    cumulative = np.cumsum(variances)

    # Strategy 1: variance threshold
    var_based = np.argmax(cumulative >= variance_threshold) + 1

    # Strategy 2: elbow (second derivative of explained variance)
    deltas = np.diff(variances)
    second_derivative = np.diff(deltas)
    elbow_based = np.argmax(second_derivative < 0) + 2

    # Final pick
    n = max(min(var_based, elbow_based, max_valid), min_dim)
    return n



def reduce_dimensions(
    embeddings,
    method: str = "pca",
    n_components: int | None = None,
    pca_variance_threshold: float = 0.95,
    tsne_perplexity: int = 30,
    tsne_n_iter: int = 1000,
    max_auto_components: int = 100,
    fitted_model=None  # <--- new
) -> tuple[np.ndarray, str, int, object]:  # return model too
    print(f"Reducing dimensions using {method} with n_components={n_components}")
    
    embeddings = np.array(embeddings)
    method = method.lower()

    if fitted_model:
        reduced = fitted_model.transform(embeddings)
        return reduced, method, fitted_model.n_components_, fitted_model

    if n_components is None:
        n_components = hybrid_n_components(
            embeddings,
            variance_threshold=pca_variance_threshold,
            max_cap=max_auto_components
        )
        print(f"Auto-selected n_components={n_components} using PCA + elbow method")

    if method == "pca":
        reducer = PCA(n_components=n_components, random_state=42)
    elif method == "umap":
        reducer = umap.UMAP(n_components=n_components, random_state=42, metric="cosine")
    elif method == "tsne":
        reducer = TSNE(n_components=n_components, perplexity=tsne_perplexity, n_iter=tsne_n_iter, random_state=42)
    elif method == "svd":
        reducer = TruncatedSVD(n_components=n_components, random_state=42)
    else:
        raise ValueError(f"Unknown dimensionality reduction method: '{method}'")

    reduced = reducer.fit_transform(embeddings)
    return reduced, method, n_components, reducer

