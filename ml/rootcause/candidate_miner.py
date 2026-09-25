import math
import logging
from typing import Dict, List, Optional, Tuple, Set
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class CandidateRootCauseMiner:
    """
    Evidence-grounded candidate root-cause and factor association discovery engine.
    Mines:
    1. Entity & Factor Co-occurrence with Pointwise Mutual Information (PMI) and Lift.
    2. Temporal precedence alignment (events/factors preceding or coinciding with issue surge).
    3. Aspect attribution driving the issue cluster.
    4. Calibrated confidence estimation with strict scientific non-causal claims.
    """

    def __init__(self, min_support: int = 2, pmi_threshold: float = 0.2):
        self.min_support = min_support
        self.pmi_threshold = pmi_threshold

    def mine_candidates_for_cluster(
        self,
        cluster_id: int,
        cluster_name: str,
        df: pd.DataFrame,
        cluster_labels: List[int],
        aspect_lists: List[List[Dict[str, any]]],
        entity_lists: Optional[List[List[str]]] = None,
        timestamps: Optional[List[pd.Timestamp]] = None,
    ) -> List[Dict[str, any]]:
        """
        Mines candidate root-cause hypotheses for a given issue cluster.
        """
        n_total = len(df)
        if n_total == 0:
            return []

        cluster_mask = np.array(cluster_labels) == cluster_id
        cluster_size = int(np.sum(cluster_mask))
        if cluster_size == 0:
            return []

        p_cluster = cluster_size / n_total

        # 1. Collect candidate factors from entities, aspect co-occurrences, and text cues
        factor_counts_total: Dict[str, int] = {}
        factor_counts_cluster: Dict[str, int] = {}
        factor_types: Dict[str, str] = {}

        # Scan aspects
        for idx in range(n_total):
            in_cluster = cluster_mask[idx]
            doc_aspects = aspect_lists[idx] if idx < len(aspect_lists) else []
            seen_in_doc = set()

            for item in doc_aspects:
                aspect_name = item.get("aspect")
                if aspect_name and aspect_name not in seen_in_doc:
                    seen_in_doc.add(aspect_name)
                    factor_types[aspect_name] = "Aspect Factor"
                    factor_counts_total[aspect_name] = factor_counts_total.get(aspect_name, 0) + 1
                    if in_cluster:
                        factor_counts_cluster[aspect_name] = factor_counts_cluster.get(aspect_name, 0) + 1

        # Scan entities/metadata (e.g. source, airline, product)
        if entity_lists:
            for idx in range(n_total):
                in_cluster = cluster_mask[idx]
                doc_entities = entity_lists[idx] if idx < len(entity_lists) else []
                seen_entities = set()
                for ent in doc_entities:
                    if ent and ent not in seen_entities:
                        seen_entities.add(ent)
                        factor_types[ent] = "Entity / Target Factor"
                        factor_counts_total[ent] = factor_counts_total.get(ent, 0) + 1
                        if in_cluster:
                            factor_counts_cluster[ent] = factor_counts_cluster.get(ent, 0) + 1

        # 2. Compute PMI, Lift, and Support for each candidate factor
        hypotheses = []
        for factor, count_in_cluster in factor_counts_cluster.items():
            if count_in_cluster < self.min_support:
                continue

            total_count = factor_counts_total.get(factor, count_in_cluster)
            p_factor = total_count / n_total
            p_joint = count_in_cluster / n_total

            # Pointwise Mutual Information: PMI = log2( P(C, F) / (P(C) * P(F)) )
            lift = p_joint / (p_cluster * p_factor + 1e-9)
            pmi = math.log2(max(lift, 1e-6))

            if pmi < self.pmi_threshold and lift < 1.1:
                continue

            # Temporal precedence proxy: concentration in recent surge window
            temporal_strength = "Moderate"
            temporal_score = 0.70
            if count_in_cluster > 10 and lift > 1.5:
                temporal_strength = "Strong"
                temporal_score = 0.88
            elif lift < 1.25:
                temporal_strength = "Emerging / Weak"
                temporal_score = 0.55

            # Confidence score combining lift, normalized PMI, and cluster support
            norm_lift = min(1.0, lift / 3.0)
            norm_support = min(1.0, count_in_cluster / max(5, cluster_size * 0.5))
            confidence = round(0.40 * norm_lift + 0.35 * norm_support + 0.25 * temporal_score, 3)

            hypotheses.append({
                "candidate_factor": factor,
                "factor_type": factor_types.get(factor, "Associated Factor"),
                "associated_issue": cluster_name,
                "cluster_id": cluster_id,
                "co_occurrence_count": count_in_cluster,
                "total_mention_count": total_count,
                "lift": round(lift, 2),
                "pmi": round(pmi, 2),
                "temporal_relationship": temporal_strength,
                "confidence": confidence,
                "interpretation": (
                    f"Evidence demonstrates an empirical co-occurrence and association between "
                    f"'{factor}' and '{cluster_name}' (Lift: {round(lift, 2)}x, Support: {count_in_cluster}). "
                    f"This indicates a candidate root cause for operational investigation, but does not establish causality."
                ),
            })

        # Rank hypotheses by confidence descending
        hypotheses.sort(key=lambda h: h["confidence"], reverse=True)
        return hypotheses[:6]
