import pytest
import pandas as pd
import numpy as np
from ml.temporal.temporal_analyzer import TemporalAnalyzer


def test_zero_count_period_retention():
    analyzer = TemporalAnalyzer()
    # 2 timestamps separated by 4 days
    t0 = pd.Timestamp("2026-03-01 12:00:00", tz="UTC")
    t1 = pd.Timestamp("2026-03-05 12:00:00", tz="UTC")
    timestamps = [t0, t1]
    cluster_labels = [0, 0]
    sentiments = ["negative", "negative"]

    df_ts = analyzer.build_time_series(timestamps, cluster_labels, sentiments, freq="D")
    # Must contain all 5 days: March 1, 2, 3, 4, 5
    assert len(df_ts) == 5
    # Middle days (March 2, 3, 4) must have total_count == 0
    zero_days = df_ts[df_ts["total_count"] == 0]
    assert len(zero_days) == 3


def test_no_lookahead_leakage():
    analyzer = TemporalAnalyzer(baseline_window_days=3)
    # Series with a massive spike on day 5
    timestamps = [
        pd.Timestamp(f"2026-03-0{i+1} 10:00:00", tz="UTC")
        for i in range(5)
    ]
    # Counts: Day 1: 5, Day 2: 5, Day 3: 5, Day 4: 5, Day 5: 500 (spike)
    expanded_ts = (
        [timestamps[0]] * 5
        + [timestamps[1]] * 5
        + [timestamps[2]] * 5
        + [timestamps[3]] * 5
        + [timestamps[4]] * 50
    )
    labels = [0] * len(expanded_ts)

    df_ts = analyzer.build_time_series(expanded_ts, labels, freq="D")
    trend = analyzer.analyze_cluster_trend(df_ts, cluster_id=0)

    # For Day 2 (index 1) and Day 3 (index 2), the z-scores must be 0.0
    # because historical counts were consistently 5, even though Day 5 has a 50-count spike!
    history = trend["history"]
    assert history[0]["count"] == 5
    assert history[1]["z_score"] == 0.0
    assert history[2]["z_score"] == 0.0
    # Day 5 (index 4) should detect the spike
    assert history[4]["z_score"] > 2.0


def test_sparse_data_guardrail():
    analyzer = TemporalAnalyzer(min_support_for_alarm=3)
    # A single event on day 1 and 0 on day 2
    timestamps = [pd.Timestamp("2026-03-01 10:00:00", tz="UTC")]
    labels = [0]
    df_ts = analyzer.build_time_series(timestamps, labels, freq="D")
    trend = analyzer.analyze_cluster_trend(df_ts, cluster_id=0)
    assert trend["data_sufficiency"] == "SPARSE_DATA"
    assert trend["z_score_anomalies"] == 0


def test_bayesian_surge_probability_scenarios():
    # 1. Stable prevalence: 10% in baseline (10/100) and 10% in recent (10/100)
    prob_stable = TemporalAnalyzer.compute_bayesian_surge_probability(
        y_b=10, n_b=100, y_r=10, n_r=100, random_seed=42
    )
    assert prob_stable is not None
    assert 0.40 <= prob_stable <= 0.60

    # 2. Increasing prevalence: 10% in baseline (10/100) -> 35% in recent (35/100)
    prob_surge = TemporalAnalyzer.compute_bayesian_surge_probability(
        y_b=10, n_b=100, y_r=35, n_r=100, random_seed=42
    )
    assert prob_surge is not None
    assert prob_surge >= 0.95

    # 3. Decreasing prevalence: 30% in baseline (30/100) -> 5% in recent (5/100)
    prob_decline = TemporalAnalyzer.compute_bayesian_surge_probability(
        y_b=30, n_b=100, y_r=5, n_r=100, random_seed=42
    )
    assert prob_decline is not None
    assert prob_decline <= 0.05

    # 4. Total volume increases 10x, but prevalence stays constant (10% -> 10%)
    # Total source volume explosion must NOT be flagged as an issue surge
    prob_source_growth = TemporalAnalyzer.compute_bayesian_surge_probability(
        y_b=10, n_b=100, y_r=100, n_r=1000, random_seed=42
    )
    assert prob_source_growth is not None
    assert 0.35 <= prob_source_growth <= 0.60

    # 5. Invalid / empty cases return None
    assert TemporalAnalyzer.compute_bayesian_surge_probability(0, 0, 5, 10) is None
    assert TemporalAnalyzer.compute_bayesian_surge_probability(5, 10, 0, 0) is None
    assert TemporalAnalyzer.compute_bayesian_surge_probability(15, 10, 5, 10) is None  # y > n

    # 6. Reproducibility with fixed seed
    p1 = TemporalAnalyzer.compute_bayesian_surge_probability(10, 100, 20, 100, random_seed=123)
    p2 = TemporalAnalyzer.compute_bayesian_surge_probability(10, 100, 20, 100, random_seed=123)
    assert p1 == p2
