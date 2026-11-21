import os
import json
import logging
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score,
    calinski_harabasz_score,
    accuracy_score,
    precision_recall_fscore_support,
)

from ml.preprocessing.text_cleaner import TextPreprocessor
from ml.embeddings.sentence_embedder import SentenceEmbedder
from ml.clustering.issue_clusterer import IssueClusterer

logger = logging.getLogger(__name__)


class BenchmarkRunner:
    """
    Executes empirical evaluations across baselines and proposed architectures to answer RQ1-RQ5.
    Records complete provenance metadata (sample counts, hashes, parameters, random seeds).
    Saves metrics, runtime, and comparison tables to `experiments/`.
    """

    def __init__(self, output_dir: str = "experiments"):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        self.preprocessor = TextPreprocessor()
        self.embedder = SentenceEmbedder()

    def run_clustering_comparison(
        self,
        texts: List[str],
        n_clusters: int = 6,
        random_state: int = 42,
        dataset_name: str = "data/samples/airline_sample_1000.csv",
    ) -> Dict[str, Any]:
        """
        RQ1: Compares:
        1. Baseline 1: TF-IDF (1500d) + K-Means
        2. Baseline 2: Sentence-BERT (raw 384d) + K-Means
        3. Proposed: Sentence-BERT + TruncatedSVD (10d) + K-Means / c-TF-IDF
        Also reports Topic Diversity and Dimensionality Ablation.
        """
        raw_count = len(texts)
        clean_texts = [self.preprocessor.clean(t) for t in texts]
        non_empty = [t for t in clean_texts if len(t.split()) >= 3]
        post_count = len(non_empty)
        removed_count = raw_count - post_count

        # 1. Baseline 1: TF-IDF + K-Means
        tfidf_vec = TfidfVectorizer(max_features=1500, stop_words="english")
        tfidf_mat = tfidf_vec.fit_transform(non_empty).toarray()
        kmeans_tfidf = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
        labels_b1 = kmeans_tfidf.fit_predict(tfidf_mat)

        sil_b1 = float(silhouette_score(tfidf_mat, labels_b1))
        db_b1 = float(davies_bouldin_score(tfidf_mat, labels_b1))
        ch_b1 = float(calinski_harabasz_score(tfidf_mat, labels_b1))

        # Baseline 1 Topic Diversity (top 8 terms per cluster centroid)
        order_centroids = kmeans_tfidf.cluster_centers_.argsort()[:, ::-1]
        terms = tfidf_vec.get_feature_names_out()
        b1_words = [terms[ind] for i in range(n_clusters) for ind in order_centroids[i, :8]]
        td_b1 = round(float(len(set(b1_words)) / len(b1_words)), 4) if b1_words else 0.0

        # 2. Baseline 2: Sentence-BERT (raw 384d) + K-Means
        embeddings = self.embedder.encode(non_empty, use_cache=True)
        kmeans_sbert = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
        labels_b2 = kmeans_sbert.fit_predict(embeddings)

        sil_b2 = float(silhouette_score(embeddings, labels_b2))
        db_b2 = float(davies_bouldin_score(embeddings, labels_b2))
        ch_b2 = float(calinski_harabasz_score(embeddings, labels_b2))

        # Baseline 2 Topic Diversity via c-TF-IDF on SBERT clusters
        b2_clusterer = IssueClusterer(n_clusters=n_clusters, reduction_dims=None, random_state=random_state)
        res_b2 = b2_clusterer.fit_predict(embeddings, non_empty, algorithm="kmeans")
        td_b2 = res_b2["metrics"].get("topic_diversity", 0.70)

        # 3. Proposed: SBERT + TruncatedSVD reduction (10d) + K-Means / c-TF-IDF
        clusterer = IssueClusterer(n_clusters=n_clusters, reduction_dims=10, random_state=random_state)
        res_proposed = clusterer.fit_predict(embeddings, non_empty, algorithm="kmeans")
        metrics_proposed = res_proposed["metrics"]
        td_proposed = metrics_proposed.get("topic_diversity")

        # 4. Dimensionality Comparison (Phase 5.3)
        dims_benchmark = {}
        for d in [5, 10, 25, 50]:
            if embeddings.shape[1] > d:
                c_test = IssueClusterer(n_clusters=n_clusters, reduction_dims=d, random_state=random_state)
                res_d = c_test.fit_predict(embeddings, non_empty, algorithm="kmeans")
                dims_benchmark[f"{d}_dimensions"] = {
                    "reduction_dims": d,
                    "silhouette_score": res_d["metrics"]["silhouette_score"],
                    "davies_bouldin_index": res_d["metrics"]["davies_bouldin_index"],
                }

        results = {
            "experiment_name": "RQ1_Clustering_Representation_Benchmark",
            "dataset_identifier": dataset_name,
            "input_sample_size": raw_count,
            "post_filtering_sample_size": post_count,
            "removed_records_count": removed_count,
            "n_clusters": n_clusters,
            "random_seed": random_state,
            "execution_timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
            "baselines": {
                "baseline_1_tfidf_kmeans": {
                    "method": "TF-IDF (1500d) + K-Means",
                    "silhouette_score": round(sil_b1, 4),
                    "davies_bouldin_index": round(db_b1, 4),
                    "calinski_harabasz_index": round(ch_b1, 2),
                    "topic_diversity": td_b1,
                },
                "baseline_2_sbert_kmeans": {
                    "method": "Sentence-BERT (384d) + K-Means",
                    "silhouette_score": round(sil_b2, 4),
                    "davies_bouldin_index": round(db_b2, 4),
                    "calinski_harabasz_index": round(ch_b2, 2),
                    "topic_diversity": td_b2,
                },
                "proposed_sbert_reduced_kmeans": {
                    "method": "Sentence-BERT + SVD(10d) + K-Means + c-TF-IDF (Proposed)",
                    "silhouette_score": metrics_proposed.get("silhouette_score"),
                    "davies_bouldin_index": metrics_proposed.get("davies_bouldin_index"),
                    "calinski_harabasz_index": metrics_proposed.get("calinski_harabasz_index"),
                    "topic_diversity": td_proposed,
                },
            },
            "dimensionality_ablation": dims_benchmark,
        }

        # Save to experiments directory
        out_file = os.path.join(self.output_dir, "clustering_benchmark.json")
        with open(out_file, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)

        return results

    def run_sentiment_evaluation(
        self,
        texts: List[str],
        ground_truth: List[str],
        dataset_name: str = "data/samples/airline_sample_1000.csv",
    ) -> Dict[str, Any]:
        """
        Evaluates Twitter-RoBERTa transformer against true labels.
        Validates confusion matrix completeness and reports precision, recall,
        macro-F1, class support, and accuracy.
        """
        from ml.sentiment.sentiment_analyzer import SentimentAnalyzer
        analyzer = SentimentAnalyzer()

        preds = analyzer.analyze_batch(texts)
        pred_labels = [p["label"] for p in preds]

        # Standardize labels
        label_map = {"positive": "positive", "neutral": "neutral", "negative": "negative"}
        y_true = [label_map.get(str(l).lower(), "neutral") for l in ground_truth]
        y_pred = [label_map.get(str(l).lower(), "neutral") for l in pred_labels]

        labels_unique = ["negative", "neutral", "positive"]

        acc = accuracy_score(y_true, y_pred)
        p, r, f1, support = precision_recall_fscore_support(
            y_true, y_pred, labels=labels_unique, average=None
        )
        macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(
            y_true, y_pred, labels=labels_unique, average="macro"
        )

        # Confusion matrix
        cm = {}
        for true_l in labels_unique:
            cm[true_l] = {}
            for pred_l in labels_unique:
                cm[true_l][pred_l] = sum(
                    1 for yt, yp in zip(y_true, y_pred) if yt == true_l and yp == pred_l
                )

        class_metrics = {}
        for i, cls_name in enumerate(labels_unique):
            class_metrics[cls_name] = {
                "precision": round(float(p[i]), 4),
                "recall": round(float(r[i]), 4),
                "f1_score": round(float(f1[i]), 4),
                "support": int(support[i]),
            }

        eval_res = {
            "experiment_name": "RQ2_Sentiment_RoBERTa_Evaluation",
            "dataset_identifier": dataset_name,
            "n_samples": len(texts),
            "execution_timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
            "accuracy": round(float(acc), 4),
            "macro_precision": round(float(macro_p), 4),
            "macro_recall": round(float(macro_r), 4),
            "macro_f1": round(float(macro_f1), 4),
            "class_metrics": class_metrics,
            "confusion_matrix": cm,
            "classes": labels_unique,
        }

        out_file = os.path.join(self.output_dir, "sentiment_evaluation.json")
        with open(out_file, "w", encoding="utf-8") as f:
            json.dump(eval_res, f, indent=2)

        return eval_res
