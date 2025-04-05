"""
ForqanTopic core module.

Defines the ForqanTopic class, which handles end-to-end topic modeling:
- Embedding
- Dimensionality Reduction
- Clustering
- Topic Naming via LLM
- Topic Assignment (LLM or Cluster-based)
"""

import fireducks.pandas as pd
from typing import List


class ForqanTopic:
    """
    ForqanTopic is a modular pipeline for unsupervised topic modeling.

    It transforms raw texts into clusters of semantically similar topics,
    using embedding models, dimensionality reduction, clustering, and LLM-based naming.
    """

    def __init__(
        self,
        embedding_model="text-embedding-3-small",
        llm_model="gpt-4o-mini",
        reducer_method=None,
        clustering_method=None,
        n_components=None,
        n_clusters=None,
    ):
        """
        Initializes the ForqanTopic pipeline with optional customization.

        Args:
            embedding_model: Model name used to generate text embeddings.
            llm_model: LLM name used for naming topics and zero-shot prediction.
            reducer_method: Method used for dimensionality reduction (e.g. "pca", "umap").
            clustering_method: Clustering algorithm (e.g. "kmeans", "hdbscan").
            n_components: Number of dimensions to reduce to (auto if None).
            n_clusters: Number of clusters (auto if None and supported by method).
        """
        self.embedding_model = embedding_model
        self.llm_model = llm_model
        self.reducer_method = reducer_method
        self.clustering_method = clustering_method
        self.n_components = n_components
        self.n_clusters = n_clusters

    def fit(self, texts: List[str]) -> dict[int, str]:
        """
        Runs the full topic modeling pipeline on the input texts.

        Steps:
        - Clean text
        - Generate embeddings
        - Reduce dimensions
        - Cluster
        - Sample representative texts
        - Name clusters with LLM
        - Compute cluster centers

        Returns:
            A dictionary with:
              - "cluster_names": topic names for each original cluster
              - "llm_names": refined topic space from LLM reorganization
        """
        from forqantopic.preprocessing import clean_texts
        from forqantopic.embedding import get_embeddings
        from forqantopic.reduction import reduce_dimensions
        from forqantopic.clustering import cluster_embeddings
        from forqantopic.sampling import sample_near_cluster_centers
        from forqantopic.topic_namer import name_clusters_with_llm
        from forqantopic.evaluation import compute_cluster_centers

        self.reducer_method = self.reducer_method or "pca"
        self.clustering_method = self.clustering_method or "kmeans"

        self.cleaned_texts = clean_texts(texts)
        self.embeddings = get_embeddings(self.cleaned_texts, model_name=self.embedding_model)
        self.reduced_embeddings, self.reducer_method, self.n_components, self.reducer_model = reduce_dimensions(
            self.embeddings, self.reducer_method, self.n_components
        )
        self.labels, self.cluster_model = cluster_embeddings(
            self.reduced_embeddings, self.clustering_method, self.n_clusters
        )

        self.cluster_texts = sample_near_cluster_centers(self.reduced_embeddings, self.labels, self.cleaned_texts)
        self.topic_names, self.llm_labels = name_clusters_with_llm(self.cluster_texts, model_name=self.llm_model)
        self.cluster_centers = compute_cluster_centers(self.reduced_embeddings, self.labels)

        return {"cluster_names": self.topic_names, "llm_names": self.llm_labels}

    def _check_fitted(self):
        if not hasattr(self, "topic_names") or not hasattr(self, "llm_labels"):
            raise ValueError("You must call fit() before using this method.")

    def transform_using_llm(self) -> pd.DataFrame:
        """
        Assigns each text to a topic name using the LLM (zero-shot classification).

        Returns:
            DataFrame with columns: 'text', 'topic'
        """
        self._check_fitted()
        topics = [self.predict_llm_label(text) for text in self.cleaned_texts]
        return pd.DataFrame({"text": self.cleaned_texts, "topic": topics})

    def fit_transform_using_llm(self, texts: List[str]) -> pd.DataFrame:
        """
        Fits the model and returns topic labels using LLM-based classification.
        """
        self.fit(texts)
        return self.transform_using_llm()

    def predict_llm_label(self, text: str) -> str:
        """
        Zero-shot topic classification using the LLM.

        Args:
            text: A raw input string.

        Returns:
            The most semantically relevant topic name based on LLM reasoning.
        """
        from litellm import completion

        prompt = f"""
        You are an intent classifier.
        Given a text, classify it into one of these topics:

        Topics:
        {", ".join([v.strip() for v in self.topic_names.values()])}

        Text:
        {text}

        Return only the topic name.
        """
        response = completion(model=self.llm_model, messages=[{"role": "user", "content": prompt}])
        return response['choices'][0]['message']['content'].strip()

    def fit_transform_using_clusters(self, texts: List[str]) -> pd.DataFrame:
        """
        Fits the model and assigns topic labels based on clustering only.
        """
        try:
            self._check_fitted()
        except ValueError:
            self.fit(texts)

        return self.transform()

    def transform(self) -> pd.DataFrame:
        """
        Assigns a cluster-based topic to each cleaned input text.

        Returns:
            DataFrame with columns: 'text', 'topic'
        """
        self._check_fitted()
        from sklearn.metrics.pairwise import cosine_similarity

        results = []
        for r in self.reduced_embeddings:
            sims = {cid: cosine_similarity([r], [center])[0][0] for cid, center in self.cluster_centers.items()}
            best_cluster = max(sims, key=sims.get)
            label_name = self.topic_names.get(best_cluster, "Unknown")
            results.append(label_name)

        return pd.DataFrame({"text": self.cleaned_texts, "topic": results})

    def predict_cluster_label(self, text: str) -> str:
        """
        Predicts the topic label using the fitted cluster model.

        Args:
            text: A raw input string.

        Returns:
            Predicted topic name based on nearest cluster assignment.
        """
        self._check_fitted()
        from forqantopic.embedding import get_embeddings
        from forqantopic.reduction import reduce_dimensions
        from sklearn.metrics.pairwise import cosine_similarity

        emb = get_embeddings([text], model_name=self.embedding_model)
        red, _, _, _ = reduce_dimensions(
            emb,
            method=self.reducer_method,
            n_components=self.n_components,
            fitted_model=self.reducer_model
        )

        if self.clustering_method in {"kmeans", "gmm", "birch"} and hasattr(self, "cluster_model"):
            try:
                cluster_id = self.cluster_model.predict(red)[0]
            except Exception:
                cluster_id = None
        else:
            sims = {cid: cosine_similarity([red[0]], [center])[0][0] for cid, center in self.cluster_centers.items()}
            cluster_id = max(sims, key=sims.get)

        return self.topic_names.get(cluster_id, "Unknown")

    def visualize(self):
        """
        Visualizes the reduced embeddings and cluster assignments using UMAP scatter plot.
        """
        self._check_fitted()
        from forqantopic.visualization import plot_umap_scatter

        plot_umap_scatter(
            reduced_embeddings=self.reduced_embeddings,
            labels=self.labels,
            texts=self.cleaned_texts,
            topic_names=self.topic_names
        )

    def evaluate(self, as_dict=False):
        """
        Computes clustering evaluation metrics and statistics.

        Args:
            as_dict: If True, returns dictionary. Otherwise, prints and returns.

        Returns:
            Dictionary with evaluation scores and cluster metadata.
        """
        self._check_fitted()
        from forqantopic.evaluation import (
            silhouette_validation,
            topic_coherence,
            inter_topic_distance,
            calinski_harabasz,
            davies_bouldin
        )

        coherence_scores = topic_coherence(self.reduced_embeddings, self.labels)
        distance_df = inter_topic_distance(self.cluster_centers)

        pretty = {
            "silhouette_score": round(float(silhouette_validation(self.reduced_embeddings, self.labels)), 3),
            "calinski_harabasz_score": round(float(calinski_harabasz(self.reduced_embeddings, self.labels)), 3),
            "davies_bouldin_score": round(float(davies_bouldin(self.reduced_embeddings, self.labels)), 3),
            "topic_coherence": {int(k): round(float(v), 3) for k, v in coherence_scores.items()},
            "inter_topic_distance": distance_df.round(3).to_dict(),
            "cluster_centers": {int(k): [round(float(val), 3) for val in v] for k, v in self.cluster_centers.items()}
        }

        if as_dict:
            return pretty
        else:
            import pprint
            pprint.pprint(pretty)
            return pretty

    def reset(self):
        """
        Clears all fitted artifacts to re-fit from scratch.
        """
        for attr in [
            "cleaned_texts", "embeddings", "reduced_embeddings",
            "labels", "cluster_texts", "topic_names", "llm_labels",
            "cluster_centers", "cluster_model", "reducer_model",
        ]:
            if hasattr(self, attr):
                delattr(self, attr)
