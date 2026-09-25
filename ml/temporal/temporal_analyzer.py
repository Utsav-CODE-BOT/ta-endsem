import logging
from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


class TemporalAnalyzer:
    """
    Temporal analytics engine for customer issue time series.
    Calculates:
    - Time-window frequencies (daily/weekly/hourly)
    - Moving averages and percentage changes
    - Statistical anomaly and change-point detection:
      1. Rolling Z-score
      2. EWMA (Exponentially Weighted Moving Average) divergence
      3. CUSUM (Cumulative Sum Control Chart) change-point detection
    """

    def __init__(
        self,
        recent_window_days: int = 2,
        baseline_window_days: int = 5,
        z_score_threshold: float = 2.0,
        ewma_alpha: float = 0.3,
    ):
        self.recent_window_days = recent_window_days
        self.baseline_window_days = baseline_window_days
        self.z_score_threshold = z_score_threshold
        self.ewma_alpha = ewma_alpha

    def build_time_series(
        self,
        timestamps: List[pd.Timestamp],
        cluster_labels: List[int],
        sentiments: Optional[List[str]] = None,
        freq: str = "D",
    ) -> pd.DataFrame:
        """
        Aggregates document occurrences by time period and cluster label.
        """
        df = pd.DataFrame({
            "timestamp": pd.to_datetime(timestamps, utc=True),
            "cluster": cluster_labels,
            "sentiment": sentiments if sentiments is not None else ["neutral"] * len(timestamps),
        })

        df["period"] = df["timestamp"].dt.to_period(freq).dt.to_timestamp()
        grouped = df.groupby(["period", "cluster"]).agg(
            total_count=("cluster", "count"),
            negative_count=("sentiment", lambda s: (s == "negative").sum()),
        ).reset_index()

        grouped["negative_ratio"] = grouped["negative_count"] / grouped["total_count"]
        return grouped

    def analyze_cluster_trend(
        self,
        series_df: pd.DataFrame,
        cluster_id: int,
    ) -> Dict[str, any]:
        """
        Analyzes trend, growth rates, EWMA, rolling Z-score, and CUSUM for a specific cluster.
        """
        sub = series_df[series_df["cluster"] == cluster_id].sort_values("period").copy()

        if sub.empty:
            return {
                "cluster_id": cluster_id,
                "history": [],
                "recent_volume": 0,
                "baseline_volume": 0,
                "growth_rate": 0.0,
                "z_score_anomalies": 0,
                "cusum_change_detected": False,
                "ewma_divergence": 0.0,
            }

        counts = sub["total_count"].values
        periods = [p.strftime("%Y-%m-%d") for p in sub["period"]]
        neg_ratios = sub["negative_ratio"].values.tolist()

        # 1. Rolling statistics & Z-Score
        z_scores = []
        for i in range(len(counts)):
            if i == 0:
                z_scores.append(0.0)
            else:
                hist = counts[:i]
                mean_val = float(np.mean(hist))
                std_val = float(np.std(hist))
                z = (counts[i] - mean_val) / (std_val + 1e-5)
                z_scores.append(round(z, 2))

        # 2. EWMA (Exponentially Weighted Moving Average)
        ewma_vals = []
        cur_ewma = float(counts[0]) if len(counts) > 0 else 0.0
        for val in counts:
            cur_ewma = self.ewma_alpha * val + (1.0 - self.ewma_alpha) * cur_ewma
            ewma_vals.append(round(cur_ewma, 2))

        # 3. CUSUM Change-Point Detection
        # S_t^+ = max(0, S_{t-1}^+ + (x_t - mu - k))
        cusum_detected = False
        if len(counts) >= 3:
            mean_base = np.mean(counts)
            k = 0.5 * (np.std(counts) + 1e-5)
            h = 4.0 * (np.std(counts) + 1e-5)
            s_pos = 0.0
            for val in counts:
                s_pos = max(0.0, s_pos + (val - mean_base - k))
                if s_pos > h:
                    cusum_detected = True
                    break

        # 4. Recent vs Baseline Growth Rate
        n_periods = len(counts)
        if n_periods >= (self.recent_window_days + 1):
            recent = np.sum(counts[-self.recent_window_days:])
            baseline = np.sum(counts[: -self.recent_window_days])
            base_norm = (baseline / max(1, n_periods - self.recent_window_days)) * self.recent_window_days
            growth_rate = ((recent - base_norm) / (base_norm + 1e-5)) * 100.0
        else:
            recent = np.sum(counts)
            baseline = 0
            growth_rate = 0.0

        history = []
        for i in range(len(periods)):
            history.append({
                "date": periods[i],
                "count": int(counts[i]),
                "z_score": float(z_scores[i]),
                "ewma": float(ewma_vals[i]),
                "negative_ratio": round(float(neg_ratios[i]), 3),
            })

        latest_z = z_scores[-1] if z_scores else 0.0
        latest_ewma_div = (counts[-1] - ewma_vals[-1]) / (ewma_vals[-1] + 1e-5) if ewma_vals else 0.0

        return {
            "cluster_id": cluster_id,
            "history": history,
            "recent_volume": int(recent),
            "baseline_volume": int(baseline),
            "growth_rate": round(float(growth_rate), 2),
            "latest_z_score": float(latest_z),
            "z_score_anomalies": sum(1 for z in z_scores if z >= self.z_score_threshold),
            "cusum_change_detected": cusum_detected,
            "ewma_divergence": round(float(latest_ewma_div), 3),
        }
