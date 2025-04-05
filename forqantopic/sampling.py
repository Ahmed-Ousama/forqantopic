def sample_near_cluster_centers(reduced_embeddings, labels, original_texts, n_samples=3):
    from collections import defaultdict
    import numpy as np

    print(f"labels {labels}")
    
    cluster_texts = defaultdict(list)
    embeddings = np.array(reduced_embeddings)
    labels = np.array(labels).flatten()  # <-- fix here

    for cluster_id in set(labels):
        if cluster_id == -1:
            continue  # Skip noise

        indices = [i for i, label in enumerate(labels) if label == cluster_id]
        cluster_points = embeddings[indices]
        cluster_center = np.mean(cluster_points, axis=0)

        distances = np.linalg.norm(cluster_points - cluster_center, axis=1)
        top_indices = np.argsort(distances)[:n_samples]
        selected_texts = [original_texts[indices[i]] for i in top_indices]

        cluster_texts[cluster_id] = selected_texts

    return cluster_texts

