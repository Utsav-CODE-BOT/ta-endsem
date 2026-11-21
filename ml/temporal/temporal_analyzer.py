import math
import logging
from typing import Dict, List, Optional, Tuple, Any
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


class TemporalAnalyzer:
    """
    Temporal analytics engine for customer issue time series.
    Provides:
    1. Calendar-complete time-series construction (preserving explicit zero-count periods).
    2. Strict prior-only rolling baseline statistics (no future leakage).
    3. CUSUM (Cumulative Sum Control Chart) change-point detection calibrated strictly on historical baseline.
    4. EWMA (Exponentially Weighted Moving Average) with explicit initialization.
    5. Sparse-data guardrails preventing spurious alarms on low support counts.
    6. Bayesian Beta-Binomial surge probability estimation.
    """

    def __init__(
        self,
        recent_window_days: int = 2,
        baseline_window_days: int = 5,
        z_score_threshold: float = 2.0,
        ewma_alpha: float = 0.3,
        min_support_for_alarm: int = 3,
    ):
        self.recent_window_days = recent_window_days
        self.baseline_window_days = baseline_window_days
        self.z_score_threshold = z_score_threshold
        self.ewma_alpha = ewma_alpha
        self.min_support_for_alarm = min_support_for_alarm

    def build_time_series(
        self,
        timestamps: List[pd.Timestamp],
        cluster_labels: List[int],
        sentiments: Optional[List[str]] = None,
        freq: str = "D",
    ) -> pd.DataFrame:
        """
        Aggregates document occurrences by time period and cluster label.
        Preserves calendar-complete zero-count periods between global start and end dates.
        """
        valid_indices = [
            i for i, t in enumerate(timestamps)
            if pd.notna(t) and t is not pd.NaT
        ]

        if not valid_indices:
            return pd.DataFrame(columns=["period", "cluster", "total_count", "negative_count", "negative_ratio"])

        clean_timestamps = [pd.to_datetime(timestamps[i], utc=True) for i in valid_indices]
        clean_clusters = [cluster_labels[i] for i in valid_indices]
        clean_sentiments = [
            sentiments[i] if (sentiments and i < len(sentiments)) else "neutral"
            for i in valid_indices
        ]

        df = pd.DataFrame({
            "timestamp": clean_timestamps,
            "cluster": clean_clusters,
            "sentiment": clean_sentiments,
        })

        # Normalize to period start
        df["period"] = df["timestamp"].dt.floor(freq)

        # Unique clusters and full date range
        unique_clusters = sorted(list(set(clean_clusters)))
        min_date = df["period"].min()
        max_date = df["period"].max()

        # Generate full date grid
        full_dates = pd.date_range(start=min_date, end=max_date, freq=freq, tz="UTC")

        # Multi-index product of all dates and clusters to guarantee ZERO-COUNT retention
        grid = pd.MultiIndex.from_product(
            [full_dates, unique_clusters],
            names=["period", "cluster"]
        ).to_frame().reset_index(drop=True)

        grouped = df.groupby(["period", "cluster"]).agg(
            total_count=("cluster", "count"),
            negative_count=("sentiment", lambda s: int((s == "negative").sum())),
        ).reset_index()

        merged = pd.merge(grid, grouped, on=["period", "cluster"], how="left")
        merged["total_count"] = merged["total_count"].fillna(0).astype(int)
        merged["negative_count"] = merged["negative_count"].fillna(0).astype(int)
        merged["negative_ratio"] = np.where(
            merged["total_count"] > 0,
            merged["negative_count"] / merged["total_count"],
            0.0
        )

        return merged.sort_values(["cluster", "period"]).reset_index(drop=True)

    def analyze_cluster_trend(
        self,
        series_df: pd.DataFrame,
        cluster_id: int,
    ) -> Dict[str, Any]:
        """
        Analyzes trend, growth rates, EWMA, rolling Z-score, and CUSUM for a specific cluster.
        Enforces prior-only rolling windows with zero look-ahead bias.
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
                "cusum_change_date": None,
                "ewma_divergence": 0.0,
                "surge_probability": 0.50,
                "data_sufficiency": "INSUFFICIENT_DATA",
            }

        counts = sub["total_count"].values.astype(float)
        periods = [p.strftime("%Y-%m-%d") for p in sub["period"]]
        neg_ratios = sub["negative_ratio"].values.tolist()
        n_periods = len(counts)

        # 1. Prior-Only Rolling Z-Score (Window W = baseline_window_days)
        z_scores = []
        for i in range(n_periods):
            if i == 0:
                z_scores.append(0.0)
            else:
                # Use strictly prior observations up to baseline_window_days
                start_idx = max(0, i - self.baseline_window_days)
                prior_window = counts[start_idx:i]
                mean_base = float(np.mean(prior_window))
                std_base = float(np.std(prior_window))

                effective_std = max(std_base, 0.25 * max(1.0, mean_base))
                z = (counts[i] - mean_base) / effective_std
                z_scores.append(round(float(z), 2))

        # 2. EWMA (Exponentially Weighted Moving Average) with explicit past-only update
        ewma_vals = []
        cur_ewma = float(counts[0]) if n_periods > 0 else 0.0
        for val in counts:
            cur_ewma = self.ewma_alpha * val + (1.0 - self.ewma_alpha) * cur_ewma
            ewma_vals.append(round(cur_ewma, 2))

        # 3. CUSUM Change-Point Detection (Calibrated on early baseline window, strictly past-only)
        cusum_detected = False
        cusum_date = None
        # Reference baseline estimated from initial baseline window
        ref_window_len = min(n_periods, max(2, self.baseline_window_days))
        ref_counts = counts[:ref_window_len]
        ref_mean = float(np.mean(ref_counts))
        ref_std = float(np.std(ref_counts))
        if ref_std < 0.1:
            ref_std = 0.5

        slack_k = 0.5 * ref_std
        threshold_h = 4.0 * ref_std

        s_pos = 0.0
        for i, val in enumerate(counts):
            s_pos = max(0.0, s_pos + (val - ref_mean - slack_k))
            if s_pos > threshold_h and not cusum_detected:
                if val >= self.min_support_for_alarm:
                    cusum_detected = True
                    cusum_date = periods[i]

        # 4. Recent vs Baseline Growth Rate
        if n_periods >= (self.recent_window_days + 1):
            recent = float(np.sum(counts[-self.recent_window_days:]))
            baseline_slice = counts[: -self.recent_window_days]
            baseline_total = float(np.sum(baseline_slice))
            baseline_days = len(baseline_slice)

            # Expected volume in a window of size recent_window_days under the baseline rate
            expected_recent = (baseline_total / max(1, baseline_days)) * self.recent_window_days
            if expected_recent > 0:
                growth_rate = ((recent - expected_recent) / expected_recent) * 100.0
            else:
                growth_rate = 100.0 if recent >= self.min_support_for_alarm else 0.0
        else:
            recent = float(np.sum(counts))
            baseline_total = 0.0
            growth_rate = 0.0

        # 5. Bayesian Beta-Binomial Surge Probability
        # Estimates P(theta_recent > theta_baseline) using analytical normal approximation
        # of the difference of two Beta posteriors
        total_cluster_vol = float(np.sum(counts))
        data_sufficiency = "SUFFICIENT" if total_cluster_vol >= 5 else "SPARSE_DATA"

        alpha_prior, beta_prior = 1.0, 1.0
        post_alpha_rec = alpha_prior + recent
        post_beta_rec = beta_prior + max(0.0, float(np.sum(counts)) - recent)
        mean_rec = post_alpha_rec / (post_alpha_rec + post_beta_rec)
        var_rec = (post_alpha_rec * post_beta_rec) / (((post_alpha_rec + post_beta_rec) ** 2) * (post_alpha_rec + post_beta_rec + 1))

        # Expected baseline proportion over same duration
        base_rate = (self.recent_window_days / max(1, n_periods))
        diff_mean = mean_rec - base_rate
        diff_std = np.sqrt(var_rec + 1e-5)
        # Probability difference > 0 via standard normal CDF approximation
        z_surge = diff_mean / diff_std
        surge_prob = 0.5 * (1.0 + math.erf(z_surge / np.sqrt(2.0)))
        surge_prob = round(float(max(0.0, min(1.0, surge_prob))), 3)

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

        # Anomalies count respecting minimum support guardrail
        anomalies_count = sum(
            1 for i, z in enumerate(z_scores)
            if z >= self.z_score_threshold and counts[i] >= self.min_support_for_alarm
        )

        return {
            "cluster_id": cluster_id,
            "history": history,
            "recent_volume": int(recent),
            "baseline_volume": int(baseline_total),
            "growth_rate": round(float(growth_rate), 2),
            "latest_z_score": float(latest_z),
            "z_score_anomalies": anomalies_count,
            "cusum_change_detected": cusum_detected,
            "cusum_change_date": cusum_date,
            "ewma_divergence": round(float(latest_ewma_div), 3),
            "surge_probability": surge_prob,
            "data_sufficiency": data_sufficiency,
        }
