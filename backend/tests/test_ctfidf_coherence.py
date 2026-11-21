import pytest
import numpy as np
from ml.clustering.issue_clusterer import compute_ctfidf, IssueClusterer


def test_ctfidf_manual_calculation():
    """
    Test c-TF-IDF calculation against manual closed-form calculation:
    Cluster 0: "delay delay cancellation" (length = 3)
    Cluster 1: "luggage lost delay"        (length = 3)
    """
    cluster_docs = [
        "delay delay cancellation",
        "luggage lost delay",
    ]
    ctfidf_mat, vocab = compute_ctfidf(cluster_docs, ngram_range=(1, 1), stop_words=None)
    assert len(vocab) > 0
    assert ctfidf_mat.shape == (2, len(vocab))

    # A = (3 + 3) / 2 = 3.0
    # term "delay": count in C0 = 2, in C1 = 1. Total = 3.
    # IDF("delay") = log(1 + 3.0 / 3.0) = log(2.0) ~= 0.693147
    # TF("delay", C0) = 2 / 3.0
    # c-TF-IDF("delay", C0) = (2/3) * log(2) ~= 0.4621
    delay_idx = vocab.index("delay")
    expected_delay_c0 = (2.0 / 3.0) * np.log(1.0 + 3.0 / 3.0)
    assert np.isclose(ctfidf_mat[0, delay_idx], expected_delay_c0, atol=1e-4)

    # term "cancellation": count in C0 = 1, C1 = 0. Total = 1.
    # IDF = log(1 + 3.0 / 1.0) = log(4.0) ~= 1.38629
    # TF = 1 / 3.0
    # c-TF-IDF = (1/3) * log(4.0) ~= 0.4621
    cancel_idx = vocab.index("cancellation")
    expected_cancel_c0 = (1.0 / 3.0) * np.log(1.0 + 3.0 / 1.0)
    assert np.isclose(ctfidf_mat[0, cancel_idx], expected_cancel_c0, atol=1e-4)
    # in C1, cancellation count is 0 -> c-TF-IDF must be 0
    assert ctfidf_mat[1, cancel_idx] == 0.0


def test_empirical_cluster_coherence():
    clusterer = IssueClusterer(n_clusters=2, reduction_dims=5, random_state=42)
    # Cluster 1: very tight embeddings (cosine sim ~ 1.0)
    base = np.array([1.0, 0.0, 0.0, 0.0, 0.0])
    c1 = np.tile(base, (5, 1)) + np.random.normal(0, 0.001, (5, 5))
    # Cluster 2: orthogonal direction
    base2 = np.array([0.0, 1.0, 0.0, 0.0, 0.0])
    c2 = np.tile(base2, (5, 1)) + np.random.normal(0, 0.001, (5, 5))

    embeddings = np.vstack([c1, c2])
    texts = [f"sample text {i}" for i in range(10)]

    res = clusterer.fit_predict(embeddings, texts)
    profiles = res["cluster_profiles"]
    assert len(profiles) == 2

    # Coherence of very tight cluster must be close to 1.0 (> 0.95), not a hardcoded 0.78
    for p in profiles:
        assert 0.90 <= p["cluster_coherence"] <= 1.0
        assert p["coherence_reliable"] is True


def test_open_set_unknown_issue_detection():
    clusterer = IssueClusterer(n_clusters=2, reduction_dims=3, random_state=42)
    # Known clusters around [1, 0, 0] and [0, 1, 0]
    e1 = np.array([[1.0, 0.1, 0.0], [0.95, 0.05, 0.0]])
    e2 = np.array([[0.0, 0.95, 0.1], [0.05, 1.0, 0.0]])
    emb = np.vstack([e1, e2])
    texts = ["flight delay problem", "delay late arrival", "lost baggage claim", "baggage lost airport"]

    clusterer.fit_predict(emb, texts)

    # Query 1: Close to known cluster 0
    q_known = np.array([0.98, 0.08, 0.0])
    pred_known = clusterer.predict_cluster(q_known, similarity_threshold=0.5)
    assert pred_known["is_known"] is True
    assert pred_known["cluster_id"] >= 0

    # Query 2: Completely orthogonal / novel embedding [0, 0, 1]
    q_unknown = np.array([0.0, 0.0, 1.0])
    pred_unknown = clusterer.predict_cluster(q_unknown, similarity_threshold=0.5)
    assert pred_unknown["is_known"] is False
    assert pred_unknown["cluster_id"] == -1
    assert "Unknown" in pred_unknown["name"]
