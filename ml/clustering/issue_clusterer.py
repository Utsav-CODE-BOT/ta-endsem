import logging
from typing import Dict, List, Optional, Tuple, Union, Any
import numpy as np
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score

logger = logging.getLogger(__name__)


def compute_ctfidf(
    cluster_docs: List[str],
    ngram_range: Tuple[int, int] = (1, 2),
    max_features: int = 3000,
    stop_words: str = "english",
) -> Tuple[np.ndarray, List[str]]:
    """
    Computes class-based TF-IDF (c-TF-IDF) according to the mathematically principled formulation:
        TF(t, c) = count(t, c) / total_terms(c)
        IDF(t)   = log(1 + A / total_count(t))
        c-TF-IDF(t, c) = TF(t, c) * IDF(t)
    where A is the average class-document length across all classes:
        A = (1 / K) * sum_c total_terms(c)

    Returns:
        (ctfidf_matrix of shape [K, V], vocabulary_list of length V)
    """
    K = len(cluster_docs)
    if K == 0:
        return np.empty((0, 0)), []

    # Handle all-empty case
    if all(len(d.strip()) == 0 for d in cluster_docs):
        return np.zeros((K, 0)), []

    vectorizer = CountVectorizer(
        ngram_range=ngram_range,
        max_features=max_features,
        stop_words=stop_words,
    )
    try:
        count_mat = vectorizer.fit_transform(cluster_docs).toarray().astype(float)
        vocabulary = vectorizer.get_feature_names_out().tolist()
    except ValueError:
        # e.g., empty vocabulary / all stop words
        return np.zeros((K, 0)), []

    if count_mat.shape[1] == 0:
        return count_mat, []

    # Total terms per cluster (class document length)
    class_lengths = count_mat.sum(axis=1)  # shape [K]
    A = float(np.mean(class_lengths)) if K > 0 else 1.0
    if A <= 0:
        A = 1.0

    # Normalized TF: count(t, c) / class_length(c)
    # Prevent division by zero for empty clusters
    tf = np.zeros_like(count_mat)
    for c in range(K):
        c_len = class_lengths[c]
        if c_len > 0:
            tf[c, :] = count_mat[c, :] / c_len

    # Global term frequencies across all classes
    global_term_counts = count_mat.sum(axis=0)  # shape [V]

    # IDF(t) = log(1 + A / total_count(t))
    idf = np.log(1.0 + (A / (global_term_counts + 1e-9)))

    # c-TF-IDF = TF * IDF
    ctfidf_mat = tf * idf
    return ctfidf_mat, vocabulary


class IssueClusterer:
    """
    Semantic issue discovery module implementing:
    1. Dimensionality reduction (TruncatedSVD / PCA).
    2. Semantic clustering (K-Means or Agglomerative).
    3. Rigorous c-TF-IDF cluster representation and deterministic labeling.
    4. Exact empirical cluster coherence: mean cosine similarity to normalized centroid.
    5. Quantitative clustering evaluation (Silhouette, Davies-Bouldin, Calinski-Harabasz, Topic Diversity).
    6. Open-set unknown issue detection for out-of-distribution inputs.
    """

    def __init__(
        self,
        n_clusters: int = 6,
        reduction_dims: Optional[int] = 10,
        random_state: int = 42,
    ):
        self.n_clusters = n_clusters
        self.reduction_dims = reduction_dims
        self.random_state = random_state

        self.reducer_cluster: Optional[TruncatedSVD] = None
        self.reducer_2d: Optional[PCA] = None
        self.clustering_model = None
        self.centroids_: Optional[np.ndarray] = None
        self.cluster_profiles_: List[Dict[str, Any]] = []

    def fit_predict(
        self,
        embeddings: np.ndarray,
        texts: List[str],
        algorithm: str = "kmeans",
    ) -> Dict[str, Any]:
        """
        Fit clustering model on embeddings and texts.
        Supported algorithms: 'kmeans', 'agglomerative'.
        Returns cluster assignments, 2D coordinates, cluster profiles, and metrics.
        """
        n_samples = len(texts)
        if n_samples == 0:
            return {
                "labels": [],
                "coords_2d": [],
                "cluster_profiles": [],
                "metrics": self._empty_metrics(),
                "n_clusters": 0,
                "algorithm": algorithm,
            }

        effective_k = max(1, min(self.n_clusters, n_samples))

        # Ensure embeddings are 2D float array
        emb = np.asarray(embeddings, dtype=float)
        if emb.ndim == 1:
            emb = emb.reshape(1, -1)

        # 1. 2D projection for interactive scatter visualization (PCA)
        if n_samples >= 2 and emb.shape[1] >= 2:
            self.reducer_2d = PCA(n_components=2, random_state=self.random_state)
            coords_2d = self.reducer_2d.fit_transform(emb)
        else:
            coords_2d = np.zeros((n_samples, 2), dtype=float)

        # 2. Dimensionality reduction for clustering density
        if (
            self.reduction_dims is not None
            and self.reduction_dims > 0
            and emb.shape[1] > self.reduction_dims
            and n_samples > self.reduction_dims
        ):
            self.reducer_cluster = TruncatedSVD(
                n_components=self.reduction_dims,
                random_state=self.random_state,
            )
            reduced_features = self.reducer_cluster.fit_transform(emb)
        else:
            self.reducer_cluster = None
            reduced_features = emb

        # 3. Cluster formation
        if effective_k == 1:
            labels = np.zeros(n_samples, dtype=int)
            self.clustering_model = None
        elif algorithm == "agglomerative":
            self.clustering_model = AgglomerativeClustering(n_clusters=effective_k)
            labels = self.clustering_model.fit_predict(reduced_features)
        else:
            self.clustering_model = KMeans(
                n_clusters=effective_k,
                random_state=self.random_state,
                n_init=10,
            )
            labels = self.clustering_model.fit_predict(reduced_features)

        # 4. Evaluation metrics
        metrics = self._calculate_metrics(emb, labels, effective_k)

        # 5. Issue representations, actual empirical coherence & labeling via c-TF-IDF
        cluster_profiles = self._generate_cluster_profiles(
            texts=texts,
            embeddings=emb,
            labels=labels,
            coords_2d=coords_2d,
            effective_k=effective_k,
        )
        self.cluster_profiles_ = cluster_profiles

        # Add topic diversity metric if applicable
        metrics["topic_diversity"] = self._calculate_topic_diversity(cluster_profiles)

        return {
            "labels": labels.tolist(),
            "coords_2d": coords_2d.tolist(),
            "cluster_profiles": cluster_profiles,
            "metrics": metrics,
            "n_clusters": effective_k,
            "algorithm": algorithm,
        }

    def _empty_metrics(self) -> Dict[str, Optional[float]]:
        return {
            "silhouette_score": None,
            "davies_bouldin_index": None,
            "calinski_harabasz_index": None,
            "topic_diversity": None,
        }

    def _calculate_metrics(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        k: int,
    ) -> Dict[str, Optional[float]]:
        unique_labels = set(labels)
        if len(unique_labels) <= 1 or len(unique_labels) >= len(labels):
            return self._empty_metrics()

        try:
            sil = float(silhouette_score(features, labels))
            db = float(davies_bouldin_score(features, labels))
            ch = float(calinski_harabasz_score(features, labels))
            return {
                "silhouette_score": round(sil, 4),
                "davies_bouldin_index": round(db, 4),
                "calinski_harabasz_index": round(ch, 2),
                "topic_diversity": None,
            }
        except Exception as e:
            logger.warning("Error calculating clustering metrics: %s", e)
            return self._empty_metrics()

    def _calculate_topic_diversity(self, profiles: List[Dict[str, Any]], top_n: int = 10) -> Optional[float]:
        """
        Calculates topic diversity: proportion of unique words across top_n words of all clusters.
        Diversity in [0, 1]. Higher indicates non-redundant distinct clusters.
        """
        all_words = []
        for p in profiles:
            terms = p.get("top_terms", [])[:top_n]
            all_words.extend([t.lower() for t in terms])
        if not all_words:
            return None
        return round(len(set(all_words)) / len(all_words), 4)

    def _generate_cluster_profiles(
        self,
        texts: List[str],
        embeddings: np.ndarray,
        labels: np.ndarray,
        coords_2d: np.ndarray,
        effective_k: int,
    ) -> List[Dict[str, Any]]:
        profiles = []

        # Aggregate texts per cluster for c-TF-IDF
        cluster_docs = []
        for c in range(effective_k):
            members = [texts[i] for i in range(len(texts)) if labels[i] == c]
            cluster_docs.append(" ".join(members) if members else "")

        # Compute mathematically rigorous c-TF-IDF
        ctfidf_mat, feature_names = compute_ctfidf(cluster_docs, ngram_range=(1, 2), max_features=2500)
        feature_names_arr = np.array(feature_names) if feature_names else np.array([])

        centroids_list = []

        for c in range(effective_k):
            member_indices = [i for i, l in enumerate(labels) if l == c]
            count = len(member_indices)

            if count == 0:
                continue

            # 1. Top keywords from c-TF-IDF with deterministic tie-breaking
            top_terms = []
            if ctfidf_mat.shape[1] > 0 and len(feature_names_arr) > 0:
                row = ctfidf_mat[c, :]
                nonzero_indices = np.where(row > 0)[0]
                if len(nonzero_indices) > 0:
                    # Sort primarily by c-TF-IDF score descending, secondarily by term string ascending
                    scored_terms = [
                        (float(row[idx]), str(feature_names_arr[idx]))
                        for idx in nonzero_indices
                    ]
                    scored_terms.sort(key=lambda x: (-x[0], x[1]))
                    top_terms = [t for _, t in scored_terms[:8]]

            if not top_terms:
                top_terms = ["issue", "service", f"cluster_{c}"]

            # 2. Compute normalized cluster centroid and empirical cluster coherence
            cluster_embeddings = embeddings[member_indices]
            # Normalize member embeddings to unit sphere
            norms = np.linalg.norm(cluster_embeddings, axis=1, keepdims=True)
            norms = np.where(norms == 0, 1e-9, norms)
            norm_cluster_embeddings = cluster_embeddings / norms

            # Centroid is the mean of normalized embeddings, then re-normalized
            mean_vector = np.mean(norm_cluster_embeddings, axis=0, keepdims=True)
            centroid_norm = np.linalg.norm(mean_vector)
            if centroid_norm > 0:
                normalized_centroid = mean_vector / centroid_norm
            else:
                normalized_centroid = mean_vector
            centroids_list.append(normalized_centroid[0])

            # Coherence: mean cosine similarity of members to the normalized centroid
            sims_to_centroid = np.dot(norm_cluster_embeddings, normalized_centroid.T).flatten()
            if count == 1:
                # Singleton cluster: coherence is statistically uninformative
                empirical_coherence = 0.50
                coherence_reliable = False
            else:
                empirical_coherence = float(np.mean(sims_to_centroid))
                empirical_coherence = max(0.0, min(1.0, empirical_coherence))
                coherence_reliable = True

            # Representative exemplar documents closest to centroid
            sorted_local_indices = sims_to_centroid.argsort()[::-1]
            exemplars = []
            for local_idx in sorted_local_indices[:3]:
                global_idx = member_indices[local_idx]
                exemplars.append({
                    "index": global_idx,
                    "text": texts[global_idx],
                    "similarity": round(float(sims_to_centroid[local_idx]), 4),
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
                "cluster_coherence": round(empirical_coherence, 4),
                "coherence_reliable": coherence_reliable,
                "center_2d": [round(center_2d[0], 4), round(center_2d[1], 4)],
            })

        self.centroids_ = np.array(centroids_list) if centroids_list else None

        # Sort profiles by size descending
        profiles.sort(key=lambda x: x["size"], reverse=True)
        return profiles

    def predict_cluster(
        self,
        embedding: np.ndarray,
        similarity_threshold: float = 0.35,
    ) -> Dict[str, Any]:
        """
        Predicts cluster assignment for a new embedding using centroid cosine distance.
        Supports open-set unknown issue detection when maximum similarity is below threshold.
        """
        if self.centroids_ is None or len(self.centroids_) == 0 or len(self.cluster_profiles_) == 0:
            return {
                "cluster_id": -1,
                "name": "Unassigned Issue",
                "similarity": 0.0,
                "is_known": False,
            }

        vec = np.asarray(embedding, dtype=float).flatten()
        v_norm = np.linalg.norm(vec)
        if v_norm > 0:
            vec = vec / v_norm

        sims = np.dot(self.centroids_, vec)
        best_idx = int(np.argmax(sims))
        best_sim = float(sims[best_idx])

        # Match to cluster profile
        profile = None
        for p in self.cluster_profiles_:
            if p["cluster_id"] == best_idx:
                profile = p
                break
        if profile is None and self.cluster_profiles_:
            profile = self.cluster_profiles_[0]

        is_known = best_sim >= similarity_threshold
        cluster_name = profile["name"] if (profile and is_known) else "Unknown / Emerging Outlier Issue"
        matched_id = profile["cluster_id"] if (profile and is_known) else -1

        return {
            "cluster_id": matched_id,
            "name": cluster_name,
            "similarity": round(best_sim, 4),
            "is_known": is_known,
            "primary_aspect": profile.get("top_terms", ["general"])[0] if profile else "general",
        }

    def _formulate_cluster_title(self, top_terms: List[str]) -> str:
        """
        Synthesizes an informative title from distinctive c-TF-IDF keywords.
        """
        meaningful = [t.title() for t in top_terms[:3] if len(t) > 2]
        if not meaningful:
            return "General Customer Inquiries"
        return " & ".join(meaningful[:2]) + f" ({meaningful[2]})" if len(meaningful) > 2 else " & ".join(meaningful)
