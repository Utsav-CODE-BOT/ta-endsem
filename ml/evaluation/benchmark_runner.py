import os
import json
import logging
from typing import Dict, List, Optional
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score, accuracy_score, precision_recall_fscore_support

from ml.preprocessing.text_cleaner import TextPreprocessor
from ml.embeddings.sentence_embedder import SentenceEmbedder
from ml.clustering.issue_clusterer import IssueClusterer

logger = logging.getLogger(__name__)


class BenchmarkRunner:
    """
    Executes empirical evaluations across baselines and proposed architectures to answer RQ1-RQ5.
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
    ) -> Dict[str, any]:
        """
        RQ1: Compares:
        1. Baseline 1: TF-IDF + K-Means
        2. Baseline 2: Sentence-BERT (raw 384d) + K-Means
        3. Proposed: Sentence-BERT + Dimensionality Reduction (10d) + K-Means / c-TF-IDF
        """
        clean_texts = [self.preprocessor.clean(t) for t in texts]
        non_empty = [t for t in clean_texts if len(t.split()) >= 3]

        # 1. Baseline 1: TF-IDF + K-Means
        tfidf_vec = TfidfVectorizer(max_features=1500, stop_words="english")
        tfidf_mat = tfidf_vec.fit_transform(non_empty).toarray()
        kmeans_tfidf = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
        labels_b1 = kmeans_tfidf.fit_predict(tfidf_mat)

        sil_b1 = float(silhouette_score(tfidf_mat, labels_b1))
        db_b1 = float(davies_bouldin_score(tfidf_mat, labels_b1))
        ch_b1 = float(calinski_harabasz_score(tfidf_mat, labels_b1))

        # 2. Baseline 2: Sentence-BERT (raw 384d) + K-Means
        embeddings = self.embedder.encode(non_empty, use_cache=True)
        kmeans_sbert = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
        labels_b2 = kmeans_sbert.fit_predict(embeddings)

        sil_b2 = float(silhouette_score(embeddings, labels_b2))
        db_b2 = float(davies_bouldin_score(embeddings, labels_b2))
        ch_b2 = float(calinski_harabasz_score(embeddings, labels_b2))

        # 3. Proposed: SBERT + TruncatedSVD reduction + K-Means / c-TF-IDF
        clusterer = IssueClusterer(n_clusters=n_clusters, reduction_dims=10, random_state=random_state)
        res_proposed = clusterer.fit_predict(embeddings, non_empty, algorithm="kmeans")
        metrics_proposed = res_proposed["metrics"]

        results = {
            "n_samples": len(non_empty),
            "n_clusters": n_clusters,
            "baselines": {
                "baseline_1_tfidf_kmeans": {
                    "method": "TF-IDF (1500d) + K-Means",
                    "silhouette_score": round(sil_b1, 4),
                    "davies_bouldin_index": round(db_b1, 4),
                    "calinski_harabasz_index": round(ch_b1, 2),
                },
                "baseline_2_sbert_kmeans": {
                    "method": "Sentence-BERT (384d) + K-Means",
                    "silhouette_score": round(sil_b2, 4),
                    "davies_bouldin_index": round(db_b2, 4),
                    "calinski_harabasz_index": round(ch_b2, 2),
                },
                "proposed_sbert_reduced_kmeans": {
                    "method": "Sentence-BERT + SVD(10d) + K-Means + c-TF-IDF (Proposed)",
                    "silhouette_score": metrics_proposed.get("silhouette_score"),
                    "davies_bouldin_index": metrics_proposed.get("davies_bouldin_index"),
                    "calinski_harabasz_index": metrics_proposed.get("calinski_harabasz_index"),
                },
            },
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
    ) -> Dict[str, any]:
        """
        Evaluates Twitter-RoBERTa transformer against true labels.
        Reports accuracy, precision, recall, macro-F1, and confusion matrix.
        """
        from ml.sentiment.sentiment_analyzer import SentimentAnalyzer
        analyzer = SentimentAnalyzer()

        preds = analyzer.analyze_batch(texts)
        pred_labels = [p["label"] for p in preds]

        # Standardize labels
        label_map = {"positive": "positive", "neutral": "neutral", "negative": "negative"}
        y_true = [label_map.get(str(l).lower(), "neutral") for l in ground_truth]
        y_pred = [label_map.get(str(l).lower(), "neutral") for l in pred_labels]

        acc = accuracy_score(y_true, y_pred)
        p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="macro")

        labels_unique = ["negative", "neutral", "positive"]
        # Confusion matrix
        cm = {}
        for true_l in labels_unique:
            cm[true_l] = {}
            for pred_l in labels_unique:
                cm[true_l][pred_l] = sum(
                    1 for yt, yp in zip(y_true, y_pred) if yt == true_l and yp == pred_l
                )

        eval_res = {
            "n_samples": len(texts),
            "accuracy": round(float(acc), 4),
            "macro_precision": round(float(p), 4),
            "macro_recall": round(float(r), 4),
            "macro_f1": round(float(f1), 4),
            "confusion_matrix": cm,
            "classes": labels_unique,
        }

        out_file = os.path.join(self.output_dir, "sentiment_evaluation.json")
        with open(out_file, "w", encoding="utf-8") as f:
            json.dump(eval_res, f, indent=2)

        return eval_res
