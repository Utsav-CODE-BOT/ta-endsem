import pytest
import numpy as np
from ml.clustering.issue_clusterer import IssueClusterer


def test_issue_clustering():
    clusterer = IssueClusterer(n_clusters=3, reduction_dims=5)
    # 3 clusters of synthetic 20d vectors
    c1 = np.random.randn(10, 20) + 5
    c2 = np.random.randn(10, 20) - 5
    c3 = np.random.randn(10, 20)
    features = np.vstack([c1, c2, c3])
    texts = [f"baggage delay ticket issue complaint {i}" for i in range(30)]

    res = clusterer.fit_predict(features, texts, algorithm="kmeans")
    assert len(res["labels"]) == 30
    assert len(res["coords_2d"]) == 30
    assert len(res["cluster_profiles"]) <= 3
    assert res["metrics"]["silhouette_score"] is not None
    assert len(res["cluster_profiles"][0]["top_terms"]) > 0
