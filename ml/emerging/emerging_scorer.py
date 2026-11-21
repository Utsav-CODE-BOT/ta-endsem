import logging
from typing import Dict, List, Optional, Any
import numpy as np

logger = logging.getLogger(__name__)


class EmergingScorer:
    """
    Principled multi-factor emerging issue prioritization engine.
    Computes EmergingScore = sum(w_i * factor_i) where:
    - growth: normalized percentage surge over historical baseline (logistic sigmoid bounded)
    - volume: log-scaled issue count (penalizes single-instance noise)
    - recentness: ratio of occurrences in the recent observation window
    - negative_sentiment: proportion of negative customer sentiment
    - cluster_coherence: empirical semantic tightness of the issue cluster

    Crucially separates operational priority (urgency of potential impact) from
    statistical confidence (sufficiency of sample support).
    """

    DEFAULT_WEIGHTS = {
        "growth": 0.30,
        "volume": 0.20,
        "recentness": 0.20,
        "negative_sentiment": 0.20,
        "cluster_coherence": 0.10,
    }

    def __init__(self, weights: Optional[Dict[str, float]] = None):
        self.weights = dict(self.DEFAULT_WEIGHTS)
        if weights:
            # Validate non-negative weights
            for k, v in weights.items():
                if v is not None and not np.isnan(v) and v >= 0:
                    self.weights[k] = float(v)

        # Normalize weights so sum equals 1.0
        total_w = sum(self.weights.values())
        if total_w > 0:
            for k in self.weights:
                self.weights[k] /= total_w

    @staticmethod
    def _sigmoid_normalize(value: float, scale: float = 100.0) -> float:
        """Maps unbounded growth percentage (-inf, +inf) smoothly to [0, 1]."""
        if np.isnan(value) or np.isneginf(value):
            return 0.0
        if np.isposinf(value):
            return 1.0
        return float(1.0 / (1.0 + np.exp(-value / scale)))

    @staticmethod
    def assign_priority(score: float) -> str:
        if score >= 0.75:
            return "CRITICAL"
        elif score >= 0.60:
            return "HIGH"
        elif score >= 0.42:
            return "MEDIUM"
        return "LOW"

    def score_issue(
        self,
        cluster_id: int,
        cluster_name: str,
        total_volume: int,
        dataset_total_volume: int,
        growth_rate: float,
        recent_volume: int,
        negative_sentiment_ratio: float,
        cluster_coherence: float = 0.75,
        first_detected_date: Optional[str] = None,
        surge_probability: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Calculate reproducible emerging score, priority ranking, and statistical support assessment.
        Validates all inputs against boundary violations.
        """
        # Input sanitation & boundary enforcement
        total_vol = max(0, int(total_volume))
        dataset_vol = max(total_vol, int(dataset_total_volume))
        rec_vol = max(0, min(int(recent_volume), total_vol))

        clean_growth = float(growth_rate) if not (np.isnan(growth_rate) or np.isinf(growth_rate)) else 0.0
        clean_neg = float(min(1.0, max(0.0, negative_sentiment_ratio))) if not np.isnan(negative_sentiment_ratio) else 0.5
        clean_coherence = float(min(1.0, max(0.0, cluster_coherence))) if not np.isnan(cluster_coherence) else 0.5

        # 1. Growth factor: mapped via sigmoid with center at 0% growth
        norm_growth = self._sigmoid_normalize(clean_growth, scale=100.0)

        # 2. Volume factor: log-scaled against dataset size
        if dataset_vol > 0 and total_vol > 0:
            norm_volume = float(np.log1p(total_vol) / np.log1p(dataset_vol))
        else:
            norm_volume = 0.0

        # 3. Recentness factor: proportion of volume in recent window (bounded [0, 1])
        norm_recentness = float(rec_vol / max(1, total_vol))
        norm_recentness = min(1.0, max(0.0, norm_recentness))

        # 4. Negative sentiment factor: directly bounded in [0, 1]
        norm_neg = clean_neg

        # 5. Cluster coherence: bounded in [0, 1]
        norm_coherence = clean_coherence

        # Weighted composition
        score = (
            self.weights["growth"] * norm_growth
            + self.weights["volume"] * norm_volume
            + self.weights["recentness"] * norm_recentness
            + self.weights["negative_sentiment"] * norm_neg
            + self.weights["cluster_coherence"] * norm_coherence
        )
        score = round(float(min(1.0, max(0.0, score))), 4)

        priority = self.assign_priority(score)

        # Statistical support assessment: separates priority from sample confidence
        if total_vol >= 20 and rec_vol >= 5:
            confidence_level = "HIGH_STATISTICAL_SUPPORT"
            evidence_sufficient = True
        elif total_vol >= 5 and rec_vol >= 2:
            confidence_level = "MODERATE_STATISTICAL_SUPPORT"
            evidence_sufficient = True
        else:
            confidence_level = "LOW_SAMPLE_SUPPORT"
            evidence_sufficient = False

        return {
            "cluster_id": cluster_id,
            "issue_name": cluster_name,
            "emerging_score": score,
            "priority": priority,
            "growth_rate_pct": round(clean_growth, 2),
            "total_volume": total_vol,
            "recent_volume": rec_vol,
            "negative_sentiment_ratio": round(clean_neg, 4),
            "cluster_coherence": round(clean_coherence, 4),
            "first_detected_date": first_detected_date or "N/A",
            "surge_probability": surge_probability,
            "confidence_level": confidence_level,
            "evidence_sufficient": evidence_sufficient,
            "component_breakdown": {
                "norm_growth": round(norm_growth, 4),
                "norm_volume": round(norm_volume, 4),
                "norm_recentness": round(norm_recentness, 4),
                "norm_negative_sentiment": round(norm_neg, 4),
                "norm_coherence": round(norm_coherence, 4),
            },
            "weights_used": {k: round(v, 3) for k, v in self.weights.items()},
        }
