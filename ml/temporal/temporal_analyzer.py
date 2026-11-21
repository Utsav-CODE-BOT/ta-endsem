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
        surge_lambda: float = 1.0,
        monte_carlo_samples: int = 10000,
        random_seed: int = 42,
    ):
        self.recent_window_days = recent_window_days
        self.baseline_window_days = baseline_window_days
        self.z_score_threshold = z_score_threshold
        self.ewma_alpha = ewma_alpha
        self.min_support_for_alarm = min_support_for_alarm
        self.surge_lambda = surge_lambda
        self.monte_carlo_samples = monte_carlo_samples
        self.random_seed = random_seed

    @staticmethod
    def compute_bayesian_surge_probability(
        y_b: int,
        n_b: int,
        y_r: int,
        n_r: int,
        alpha: float = 1.0,
        beta: float = 1.0,
        surge_lambda: float = 1.0,
        n_samples: int = 10000,
        random_seed: int = 42,
    ) -> Optional[float]:
        """
        Calculates P(p_r > surge_lambda * p_b | D_b, D_r) via reproducible Monte Carlo sampling
        from Beta posteriors:
            p_b | D_b ~ Beta(alpha + y_b, beta + n_b - y_b)
            p_r | D_r ~ Beta(alpha + y_r, beta + n_r - y_r)
        where:
            y_b, y_r: counts of cluster C in baseline and recent periods.
            n_b, n_r: total feedback records across all clusters in baseline and recent periods.
        """
        if n_b <= 0 or n_r <= 0 or alpha <= 0 or beta <= 0 or surge_lambda <= 0 or n_samples <= 0:
            return None
        if y_b < 0 or y_r < 0 or y_b > n_b or y_r > n_r:
            return None

        rng = np.random.default_rng(random_seed)
        post_b = rng.beta(alpha + y_b, beta + (n_b - y_b), size=n_samples)
        post_r = rng.beta(alpha + y_r, beta + (n_r - y_r), size=n_samples)
        prob = float(np.mean(post_r > (surge_lambda * post_b)))
        return round(prob, 3)

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
        # Change detection monitors post-baseline periods (i >= ref_window_len) to prevent in-sample evaluation of calibration window
        for i in range(ref_window_len, n_periods):
            val = counts[i]
            s_pos = max(0.0, s_pos + (val - ref_mean - slack_k))
            if s_pos > threshold_h and not cusum_detected:
                if val >= self.min_support_for_alarm:
                    cusum_detected = True
                    cusum_date = periods[i]

        # 4. Recent vs Baseline Growth Rate & Exposure-Normalized Bayesian Surge Probability
        if "total_count" in series_df.columns:
            period_totals = series_df.groupby("period")["total_count"].sum()
        else:
            period_totals = pd.Series(dtype=float)

        if n_periods >= (self.recent_window_days + 1):
            recent_counts = counts[-self.recent_window_days:]
            baseline_counts = counts[: -self.recent_window_days]
            recent = float(np.sum(recent_counts))
            baseline_total = float(np.sum(baseline_counts))
            baseline_days = len(baseline_counts)

            # Expected volume in a window of size recent_window_days under the baseline rate
            expected_recent = (baseline_total / max(1, baseline_days)) * self.recent_window_days
            if expected_recent > 0:
                growth_rate = ((recent - expected_recent) / expected_recent) * 100.0
            else:
                growth_rate = 100.0 if recent >= self.min_support_for_alarm else 0.0

            # Bayesian Beta-Binomial Surge Probability on issue prevalence
            # Maintain pandas Series/Index without calling .values (which strips timezone and breaks alignment)
            recent_periods = sub["period"].iloc[-self.recent_window_days:]
            baseline_periods = sub["period"].iloc[: -self.recent_window_days]

            n_r = int(period_totals.reindex(recent_periods, fill_value=0).sum())
            n_b = int(period_totals.reindex(baseline_periods, fill_value=0).sum())
            y_r = int(recent)
            y_b = int(baseline_total)

            surge_prob = self.compute_bayesian_surge_probability(
                y_b=y_b,
                n_b=n_b,
                y_r=y_r,
                n_r=n_r,
                alpha=1.0,
                beta=1.0,
                surge_lambda=self.surge_lambda,
                n_samples=self.monte_carlo_samples,
                random_seed=self.random_seed,
            )
        else:
            recent = float(np.sum(counts))
            baseline_total = 0.0
            growth_rate = 0.0
            surge_prob = None

        total_cluster_vol = float(np.sum(counts))
        data_sufficiency = "SUFFICIENT" if (total_cluster_vol >= 5 and baseline_total > 0 and surge_prob is not None) else "SPARSE_DATA"

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
