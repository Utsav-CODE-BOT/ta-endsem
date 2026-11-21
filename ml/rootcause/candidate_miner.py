import math
import logging
from typing import Dict, List, Optional, Tuple, Set, Any
import numpy as np
import pandas as pd
from scipy.stats import fisher_exact

logger = logging.getLogger(__name__)


def benjamini_hochberg(p_values: List[float], alpha: float = 0.05) -> Tuple[List[float], List[bool]]:
    """
    Applies the Benjamini-Hochberg (1995) procedure to control False Discovery Rate (FDR).
    Returns (adjusted_p_values, is_significant_booleans).
    """
    m = len(p_values)
    if m == 0:
        return [], []

    indexed = sorted(enumerate(p_values), key=lambda x: x[1])
    adjusted = [1.0] * m
    significant = [False] * m

    cum_min = 1.0
    for rank in range(m, 0, -1):
        orig_idx, p = indexed[rank - 1]
        adj = min(1.0, (p * m) / rank)
        cum_min = min(cum_min, adj)
        adjusted[orig_idx] = round(cum_min, 5)
        significant[orig_idx] = cum_min <= alpha

    return adjusted, significant


class CandidateRootCauseMiner:
    """
    Evidence-grounded candidate root-cause and factor association discovery engine.
    Mines:
    1. Entity & Factor Co-occurrence with Pointwise Mutual Information (PMI), Normalized PMI (NPMI), and Lift.
    2. Statistical significance testing via Fisher's Exact Test and Benjamini-Hochberg FDR correction.
    3. Empirical temporal precedence alignment based on actual timestamps.
    4. Factor taxonomy separation (differentiating explanatory factors from tautological descriptors).
    5. Association strength scoring with explicit non-causal interpretation.
    """

    TAUTOLOGICAL_STOPWORDS = {
        "issue", "problem", "flight", "service", "customer", "plane", "airline",
        "feedback", "complaint", "experience", "hour", "day", "time", "people",
    }

    def __init__(
        self,
        min_support: int = 2,
        pmi_threshold: float = 0.1,
        alpha_significance: float = 0.05,
    ):
        self.min_support = min_support
        self.pmi_threshold = pmi_threshold
        self.alpha_significance = alpha_significance

    def _classify_factor_type(self, factor: str, cluster_name: str) -> Tuple[str, bool]:
        """
        Classifies factor into an explanatory taxonomy and checks for tautological descriptors.
        """
        factor_lower = factor.lower().strip()
        cluster_lower = cluster_name.lower().strip()

        # Tautological check: if factor is a sub-word of the cluster name or generic stopword
        is_tautological = False
        if factor_lower in cluster_lower or cluster_lower in factor_lower:
            is_tautological = True
        elif factor_lower in self.TAUTOLOGICAL_STOPWORDS:
            is_tautological = True

        # Taxonomy classification
        if any(tech in factor_lower for tech in ["50", "40", "error", "crash", "timeout", "bug", "app", "version", "v4", "v5", "api", "gateway", "database"]):
            return "SOFTWARE_OR_TECHNICAL_FAULT", is_tautological
        elif any(op in factor_lower for op in ["weather", "snow", "storm", "crew", "maintenance", "mechanical", "tarmac", "fuel", "runway", "gate"]):
            return "OPERATIONAL_OR_EXTERNAL_EVENT", is_tautological
        elif any(svc in factor_lower for svc in ["hold", "phone", "agent", "refund", "rebook", "desk", "counter", "representative", "staff", "attendant"]):
            return "SERVICE_TOUCHPOINT_FACTOR", is_tautological
        elif any(ent in factor_lower for ent in ["delta", "united", "american", "us airways", "southwest", "virgin", "jetblue", "airport", "terminal"]):
            return "ORGANIZATIONAL_OR_LOCATION_ENTITY", is_tautological
        elif is_tautological:
            return "TAUTOLOGICAL_ISSUE_DESCRIPTOR", True
        else:
            return "ASSOCIATED_SERVICE_ASPECT", False

    def _determine_temporal_precedence(
        self,
        factor_indices: List[int],
        cluster_indices: List[int],
        timestamps: Optional[List[pd.Timestamp]],
    ) -> Tuple[str, float]:
        """
        Determines empirical temporal precedence by analyzing actual timestamps of factor
        occurrences versus cluster surge occurrences.
        """
        if timestamps is None or len(timestamps) == 0:
            return "Insufficient temporal evidence (No timestamps provided)", 0.50

        # Filter valid timestamps
        valid_factor_times = [
            timestamps[i] for i in factor_indices
            if i < len(timestamps) and pd.notna(timestamps[i])
        ]
        valid_cluster_times = [
            timestamps[i] for i in cluster_indices
            if i < len(timestamps) and pd.notna(timestamps[i])
        ]

        if len(valid_factor_times) < 2 or len(valid_cluster_times) < 2:
            return "Insufficient temporal evidence (Sparse timestamp support)", 0.50

        min_factor_t = min(valid_factor_times)
        min_cluster_t = min(valid_cluster_times)

        # Calculate cluster surge start (e.g. median / 25th percentile of cluster dates)
        sorted_cluster_t = sorted(valid_cluster_times)
        cluster_surge_t = sorted_cluster_t[max(0, int(len(sorted_cluster_t) * 0.25))]

        # Count factor events before cluster surge
        factor_pre_surge = sum(1 for t in valid_factor_times if t <= cluster_surge_t)
        ratio_pre_surge = factor_pre_surge / len(valid_factor_times)

        if min_factor_t < min_cluster_t or ratio_pre_surge >= 0.50:
            return "Precedes or triggers issue surge", 0.90
        elif abs((min_factor_t - min_cluster_t).total_seconds()) <= 86400:  # within 24h
            return "Coincides with issue surge", 0.75
        else:
            return "Follows issue surge (Lagging consequence or delayed report)", 0.55

    def mine_candidates_for_cluster(
        self,
        cluster_id: int,
        cluster_name: str,
        df: pd.DataFrame,
        cluster_labels: List[int],
        aspect_lists: List[List[Dict[str, Any]]],
        entity_lists: Optional[List[List[str]]] = None,
        timestamps: Optional[List[pd.Timestamp]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Mines candidate root-cause hypotheses for a given issue cluster with statistical
        significance filtering, NPMI, and empirical temporal analysis.
        """
        n_total = len(df)
        if n_total == 0:
            return []

        cluster_mask = np.array(cluster_labels) == cluster_id
        cluster_size = int(np.sum(cluster_mask))
        if cluster_size == 0:
            return []

        cluster_indices = [i for i, m in enumerate(cluster_mask) if m]
        p_cluster = cluster_size / n_total

        # 1. Collect candidate factors
        factor_indices_map: Dict[str, List[int]] = {}
        factor_cluster_indices_map: Dict[str, List[int]] = {}

        # Scan aspects
        for idx in range(n_total):
            in_cluster = cluster_mask[idx]
            doc_aspects = aspect_lists[idx] if idx < len(aspect_lists) else []
            seen_in_doc = set()

            for item in doc_aspects:
                aspect_name = item.get("aspect")
                if aspect_name and aspect_name not in seen_in_doc:
                    seen_in_doc.add(aspect_name)
                    factor_indices_map.setdefault(aspect_name, []).append(idx)
                    if in_cluster:
                        factor_cluster_indices_map.setdefault(aspect_name, []).append(idx)

        # Scan entities
        if entity_lists:
            for idx in range(n_total):
                in_cluster = cluster_mask[idx]
                doc_entities = entity_lists[idx] if idx < len(entity_lists) else []
                seen_entities = set()
                for ent in doc_entities:
                    if ent and ent not in seen_entities:
                        seen_entities.add(ent)
                        factor_indices_map.setdefault(ent, []).append(idx)
                        if in_cluster:
                            factor_cluster_indices_map.setdefault(ent, []).append(idx)

        # 2. Compute 2x2 contingency tables and statistical metrics for candidates
        candidate_evaluations = []
        raw_p_values = []

        for factor, joint_indices in factor_cluster_indices_map.items():
            a = len(joint_indices)  # Factor Present, Issue Present
            if a < self.min_support:
                continue

            all_factor_indices = factor_indices_map.get(factor, joint_indices)
            total_factor_count = len(all_factor_indices)

            # Contingency table:
            #                   Issue      Not Issue
            # Factor Present      a           b
            # Factor Absent       c           d
            b = total_factor_count - a
            c = cluster_size - a
            d = n_total - a - b - c
            d = max(0, d)

            # Fisher's Exact Test (one-sided greater: testing positive association)
            try:
                _, p_value = fisher_exact([[a, b], [c, d]], alternative="greater")
            except Exception:
                p_value = 1.0
            raw_p_values.append(p_value)

            # Lift = P(F, I) / (P(F) * P(I))
            p_factor = total_factor_count / n_total
            p_joint = a / n_total
            lift = p_joint / (p_cluster * p_factor + 1e-9)

            # PMI = log2(P(F, I) / (P(F) * P(I)))
            pmi = math.log2(max(lift, 1e-9))

            # NPMI = PMI / (-log2(P(F, I))) -> normalized in [-1, +1]
            neg_log_joint = -math.log2(max(p_joint, 1e-9))
            npmi = pmi / neg_log_joint if neg_log_joint > 0 else 0.0
            npmi = max(-1.0, min(1.0, npmi))

            # Factor taxonomy
            factor_type, is_tautological = self._classify_factor_type(factor, cluster_name)

            # Temporal precedence
            temporal_label, temporal_score = self._determine_temporal_precedence(
                all_factor_indices, cluster_indices, timestamps
            )

            candidate_evaluations.append({
                "factor": factor,
                "a": a,
                "total_factor_count": total_factor_count,
                "lift": lift,
                "pmi": pmi,
                "npmi": npmi,
                "raw_p_value": p_value,
                "factor_type": factor_type,
                "is_tautological": is_tautological,
                "temporal_relationship": temporal_label,
                "temporal_score": temporal_score,
            })

        if not candidate_evaluations:
            return []

        # 3. Apply Benjamini-Hochberg FDR correction across candidate hypotheses
        adj_p_values, sig_mask = benjamini_hochberg(raw_p_values, alpha=self.alpha_significance)

        hypotheses = []
        for i, item in enumerate(candidate_evaluations):
            item["adjusted_p_value"] = float(adj_p_values[i])
            item["is_statistically_significant"] = bool(sig_mask[i])

            # Filter: must meet minimal association threshold or significance
            if item["pmi"] < self.pmi_threshold and item["lift"] < 1.1:
                continue

            # Heuristic association strength score in [0, 1]
            # Demotes tautological descriptors so explanatory operational factors rank higher
            norm_lift = min(1.0, item["lift"] / 3.0)
            norm_support = min(1.0, item["a"] / max(5, cluster_size * 0.4))
            norm_npmi = max(0.0, item["npmi"])
            tautology_penalty = 0.50 if item["is_tautological"] else 1.0

            raw_confidence = (
                0.30 * norm_lift
                + 0.25 * norm_support
                + 0.25 * norm_npmi
                + 0.20 * item["temporal_score"]
            ) * tautology_penalty

            association_strength = round(float(min(1.0, max(0.0, raw_confidence))), 3)

            hypotheses.append({
                "candidate_factor": item["factor"],
                "factor_type": item["factor_type"],
                "associated_issue": cluster_name,
                "cluster_id": cluster_id,
                "co_occurrence_count": item["a"],
                "total_mention_count": item["total_factor_count"],
                "lift": round(item["lift"], 2),
                "pmi": round(item["pmi"], 2),
                "npmi": round(item["npmi"], 3),
                "fisher_p_value": round(item["raw_p_value"], 5),
                "adjusted_p_value": round(item["adjusted_p_value"], 5),
                "is_statistically_significant": item["is_statistically_significant"],
                "is_tautological": item["is_tautological"],
                "temporal_relationship": item["temporal_relationship"],
                "confidence": association_strength,  # preserved for API compatibility
                "association_strength_score": association_strength,
                "interpretation": (
                    f"Statistical dependency observed between factor '{item['factor']}' and cluster '{cluster_name}' "
                    f"(Lift: {round(item['lift'], 2)}x, NPMI: {round(item['npmi'], 3)}, Support: {item['a']}, "
                    f"Fisher adj-p: {round(item['adjusted_p_value'], 4)}). "
                    f"Temporal Precedence: {item['temporal_relationship']}. "
                    f"This indicates a candidate root cause for operational investigation, but does not establish causality."
                ),
            })

        # Rank by association strength descending, favoring explanatory factors over tautological
        hypotheses.sort(key=lambda h: (not h["is_tautological"], h["association_strength_score"]), reverse=True)
        return hypotheses[:6]
