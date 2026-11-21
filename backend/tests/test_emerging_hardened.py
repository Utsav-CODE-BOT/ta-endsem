import pytest
import numpy as np
from ml.emerging.emerging_scorer import EmergingScorer


def test_emerging_scorer_input_validation():
    scorer = EmergingScorer()
    # Test with recent_volume > total_volume, NaN growth, and out-of-bound negative ratio
    res = scorer.score_issue(
        cluster_id=0,
        cluster_name="Payment Errors",
        total_volume=10,
        dataset_total_volume=50,
        growth_rate=float("nan"),
        recent_volume=20,  # exceeds total_volume
        negative_sentiment_ratio=1.5,  # exceeds 1.0
        cluster_coherence=-0.2,  # below 0.0
    )

    assert 0.0 <= res["emerging_score"] <= 1.0
    assert res["recent_volume"] <= res["total_volume"]
    assert res["component_breakdown"]["norm_recentness"] == 1.0
    assert res["component_breakdown"]["norm_negative_sentiment"] == 1.0
    assert res["component_breakdown"]["norm_coherence"] == 0.0


def test_priority_vs_confidence_separation():
    scorer = EmergingScorer()
    # Issue with very high growth (+400%) but only 2 total complaints (low support)
    res_low_support = scorer.score_issue(
        cluster_id=1,
        cluster_name="Rare Edge Case Bug",
        total_volume=2,
        dataset_total_volume=1000,
        growth_rate=400.0,
        recent_volume=2,
        negative_sentiment_ratio=1.0,
    )
    # Priority can be HIGH or MEDIUM due to growth, but confidence_level must be LOW_SAMPLE_SUPPORT
    assert res_low_support["confidence_level"] == "LOW_SAMPLE_SUPPORT"
    assert res_low_support["evidence_sufficient"] is False

    # Issue with 100 complaints and 30 recent complaints
    res_high_support = scorer.score_issue(
        cluster_id=2,
        cluster_name="Major Outage",
        total_volume=100,
        dataset_total_volume=1000,
        growth_rate=200.0,
        recent_volume=30,
        negative_sentiment_ratio=0.9,
    )
    assert res_high_support["confidence_level"] == "HIGH_STATISTICAL_SUPPORT"
    assert res_high_support["evidence_sufficient"] is True


def test_growth_monotonicity():
    scorer = EmergingScorer()
    s1 = scorer.score_issue(0, "A", 50, 500, growth_rate=50.0, recent_volume=10, negative_sentiment_ratio=0.5)
    s2 = scorer.score_issue(0, "A", 50, 500, growth_rate=150.0, recent_volume=10, negative_sentiment_ratio=0.5)
    s3 = scorer.score_issue(0, "A", 50, 500, growth_rate=300.0, recent_volume=10, negative_sentiment_ratio=0.5)
    assert s1["emerging_score"] < s2["emerging_score"] < s3["emerging_score"]
