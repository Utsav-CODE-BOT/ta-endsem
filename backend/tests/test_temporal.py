import pytest
import pandas as pd
from ml.temporal.temporal_analyzer import TemporalAnalyzer


def test_temporal_analyzer():
    ta = TemporalAnalyzer()
    dates = pd.date_range("2026-08-01", periods=10, freq="D")
    timestamps = list(dates) + list(dates[-2:]) * 5  # sudden surge in last 2 days
    clusters = [0] * len(timestamps)
    sentiments = ["negative"] * len(timestamps)

    df_ts = ta.build_time_series(timestamps, clusters, sentiments, freq="D")
    assert not df_ts.empty
    trend = ta.analyze_cluster_trend(df_ts, 0)
    assert trend["cluster_id"] == 0
    assert len(trend["history"]) > 0
    assert "growth_rate" in trend
    assert "cusum_change_detected" in trend
