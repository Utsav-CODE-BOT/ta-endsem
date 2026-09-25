import logging
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score

logger = logging.getLogger(__name__)


class IssueClusterer:
    """
    Semantic issue discovery module implementing:
    1. Dimensionality reduction (PCA / TruncatedSVD / UMAP projection).
    2. Semantic clustering (K-Means, Agglomerative, or HDBSCAN-style density).
    3. c-TF-IDF cluster representation and human-readable issue labeling.
    4. Quantitative clustering evaluation (Silhouette, Davies-Bouldin, Calinski-Harabasz).
    """

    def __init__(
        self,
        n_clusters: int = 6,
        reduction_dims: int = 10,
        random_state: int = 42,
    ):
        self.n_clusters = n_clusters
        self.reduction_dims = reduction_dims
        self.random_state = random_state

        self.reducer_cluster: Optional[TruncatedSVD] = None
        self.reducer_2d: Optional[PCA] = None
        self.clustering_model = None
        self.c_tfidf_vectorizer: Optional[TfidfVectorizer] = None

    def fit_predict(
        self,
        embeddings: np.ndarray,
        texts: List[str],
        algorithm: str = "kmeans",
    ) -> Dict[str, any]:
        """
        Fit clustering model on embeddings and texts.
        Supported algorithms: 'kmeans', 'agglomerative'.
        Returns cluster assignments, 2D coordinates, cluster profiles, and metrics.
        """
        n_samples = len(texts)
        if n_samples < self.n_clusters:
            effective_k = max(1, n_samples)
        else:
            effective_k = self.n_clusters

        # 1. 2D projection for interactive scatter visualization
        self.reducer_2d = PCA(n_components=2, random_state=self.random_state)
        coords_2d = self.reducer_2d.fit_transform(embeddings)

        # 2. Dimensionality reduction for clustering density
        if embeddings.shape[1] > self.reduction_dims and n_samples > self.reduction_dims:
            self.reducer_cluster = TruncatedSVD(
                n_components=self.reduction_dims,
                random_state=self.random_state,
            )
            reduced_features = self.reducer_cluster.fit_transform(embeddings)
        else:
            reduced_features = embeddings

        # 3. Cluster formation
        if algorithm == "agglomerative":
            self.clustering_model = AgglomerativeClustering(n_clusters=effective_k)
            labels = self.clustering_model.fit_predict(reduced_features)
        else:
            self.clustering_model = KMeans(
                n_clusters=effective_k,
                random_state=self.random_state,
                n_init=10,
            )
            labels = self.clustering_model.fit_predict(reduced_features)

        # 4. Evaluation metrics (only applicable when multiple clusters exist and > 1 sample per cluster)
        metrics = self._calculate_metrics(embeddings, labels, effective_k)

        # 5. Issue representations & labeling via c-TF-IDF
        cluster_profiles = self._generate_cluster_profiles(
            texts=texts,
            embeddings=embeddings,
            labels=labels,
            coords_2d=coords_2d,
            effective_k=effective_k,
        )

        return {
            "labels": labels.tolist(),
            "coords_2d": coords_2d.tolist(),
            "cluster_profiles": cluster_profiles,
            "metrics": metrics,
            "n_clusters": effective_k,
            "algorithm": algorithm,
        }

    def _calculate_metrics(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        k: int,
    ) -> Dict[str, Optional[float]]:
        unique_labels = set(labels)
        if len(unique_labels) <= 1 or len(unique_labels) >= len(labels):
            return {
                "silhouette_score": None,
                "davies_bouldin_index": None,
                "calinski_harabasz_index": None,
            }

        try:
            sil = float(silhouette_score(features, labels))
            db = float(davies_bouldin_score(features, labels))
            ch = float(calinski_harabasz_score(features, labels))
            return {
                "silhouette_score": round(sil, 4),
                "davies_bouldin_index": round(db, 4),
                "calinski_harabasz_index": round(ch, 2),
            }
        except Exception as e:
            logger.warning("Error calculating clustering metrics: %s", e)
            return {
                "silhouette_score": None,
                "davies_bouldin_index": None,
                "calinski_harabasz_index": None,
            }

    def _generate_cluster_profiles(
        self,
        texts: List[str],
        embeddings: np.ndarray,
        labels: np.ndarray,
        coords_2d: np.ndarray,
        effective_k: int,
    ) -> List[Dict[str, any]]:
        profiles = []

        # Aggregate texts per cluster for c-TF-IDF
        cluster_docs = []
        for c in range(effective_k):
            members = [texts[i] for i in range(len(texts)) if labels[i] == c]
            cluster_docs.append(" ".join(members) if members else "")

        vectorizer = TfidfVectorizer(
            stop_words="english",
            max_features=2500,
            ngram_range=(1, 2),
        )

        try:
            tfidf_mat = vectorizer.fit_transform(cluster_docs)
            feature_names = np.array(vectorizer.get_feature_names_out())
        except Exception:
            tfidf_mat = None
            feature_names = None

        for c in range(effective_k):
            member_indices = [i for i, l in enumerate(labels) if l == c]
            count = len(member_indices)

            if count == 0:
                continue

            # Top keywords from c-TF-IDF
            top_terms = []
            if tfidf_mat is not None and feature_names is not None:
                row = tfidf_mat.getrow(c).toarray().flatten()
                top_word_indices = row.argsort()[::-1][:8]
                top_terms = [str(feature_names[idx]) for idx in top_word_indices if row[idx] > 0]

            if not top_terms:
                top_terms = ["issue", "service", f"cluster_{c}"]

            # Compute cluster centroid
            cluster_embeddings = embeddings[member_indices]
            centroid = np.mean(cluster_embeddings, axis=0, keepdims=True)
            # Find representative exemplar documents closest to centroid
            sims = np.dot(cluster_embeddings, centroid.T).flatten()
            sorted_local_indices = sims.argsort()[::-1]

            exemplars = []
            for local_idx in sorted_local_indices[:3]:
                global_idx = member_indices[local_idx]
                exemplars.append({
                    "index": global_idx,
                    "text": texts[global_idx],
                    "similarity": round(float(sims[local_idx]), 4),
                })

            # Human-readable title generation
            title = self._formulate_cluster_title(top_terms)

            # Center coordinate in 2D
            cluster_coords = coords_2d[member_indices]
            center_2d = np.mean(cluster_coords, axis=0).tolist()

            profiles.append({
                "cluster_id": c,
                "name": title,
                "size": count,
                "percentage": round(count / len(texts) * 100, 2),
                "top_terms": top_terms,
                "exemplars": exemplars,
                "center_2d": [round(center_2d[0], 4), round(center_2d[1], 4)],
            })

        # Sort profiles by size descending
        profiles.sort(key=lambda x: x["size"], reverse=True)
        return profiles

    def _formulate_cluster_title(self, top_terms: List[str]) -> str:
        """
        Synthesizes an informative title from distinctive c-TF-IDF keywords.
        """
        meaningful = [t.title() for t in top_terms[:3] if len(t) > 2]
        if not meaningful:
            return "General Customer Inquiries"
        return " & ".join(meaningful[:2]) + f" ({meaningful[2]})" if len(meaningful) > 2 else " & ".join(meaningful)
