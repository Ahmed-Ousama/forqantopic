import numpy as np
import fireducks.pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from collections import defaultdict

def compute_cluster_centers(embeddings, labels):
    centers = {}
    labels = np.array(labels)
    embeddings = np.array(embeddings)

    for label in set(labels):
        if label == -1:
            continue  # skip noise
        points = embeddings[labels == label]
        centers[label] = np.mean(points, axis=0)

    return centers

def inter_topic_distance(centers):
    topics = list(centers.keys())
    vectors = [centers[t] for t in topics]
    sims = cosine_similarity(vectors)
    dists = 1 - sims  # convert similarity to distance
    return pd.DataFrame(dists, index=topics, columns=topics)

def topic_coherence(embeddings, labels):
    labels = np.array(labels)
    embeddings = np.array(embeddings)
    cluster_scores = {}

    for label in set(labels):
        if label == -1:
            continue
        points = embeddings[labels == label]
        if len(points) < 2:
            cluster_scores[label] = 0.0
            continue
        sim_matrix = cosine_similarity(points)
        upper_tri = sim_matrix[np.triu_indices_from(sim_matrix, k=1)]
        cluster_scores[label] = np.mean(upper_tri)

    return cluster_scores  # Dict: cluster_id -> coherence score

def silhouette_validation(embeddings, labels):
    unique_labels = set(labels)
    if len(unique_labels) <= 1 or (len(unique_labels) == 2 and -1 in unique_labels):
        return -1.0  # Not valid
    try:
        return silhouette_score(embeddings, labels)
    except Exception:
        return -1.0

def calinski_harabasz(embeddings, labels):
    try:
        return calinski_harabasz_score(embeddings, labels)
    except Exception:
        return -1.0

def davies_bouldin(embeddings, labels):
    try:
        return davies_bouldin_score(embeddings, labels)
    except Exception:
        return -1.0