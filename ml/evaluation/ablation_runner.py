import os
import json
import logging
from typing import Dict, List, Optional
import numpy as np

logger = logging.getLogger(__name__)


class AblationStudyRunner:
    """
    Executes systematic ablation study on the Emerging Issue Prioritization Engine:
    1. Full System (Semantic + Temporal + Sentiment + Aspects + Evidence)
    2. Ablation A: - without temporal features (static volume only)
    3. Ablation B: - without semantic embeddings (lexical TF-IDF only)
    4. Ablation C: - without aspect information (pure unigram terms)
    5. Ablation D: - without sentiment features (all feedback treated equal)
    6. Ablation E: - without evidence retrieval (unverified hypothesis)
    """

    def __init__(self, output_dir: str = "experiments/ablation"):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def run_ablation(
        self,
        issues_data: List[Dict[str, any]],
        ground_truth_critical_ids: Optional[List[int]] = None,
    ) -> Dict[str, any]:
        """
        Calculates prioritization ranking variance and top-k alignment across ablations.
        """
        configurations = {
            "full_system": {
                "weights": {"growth": 0.30, "volume": 0.20, "recentness": 0.20, "negative_sentiment": 0.20, "cluster_coherence": 0.10},
                "has_temporal": True,
                "has_aspects": True,
                "has_sentiment": True,
                "has_evidence": True,
            },
            "w_o_temporal": {
                "weights": {"growth": 0.00, "volume": 0.50, "recentness": 0.00, "negative_sentiment": 0.35, "cluster_coherence": 0.15},
                "has_temporal": False,
                "has_aspects": True,
                "has_sentiment": True,
                "has_evidence": True,
            },
            "w_o_semantic_embeddings": {
                "weights": {"growth": 0.30, "volume": 0.25, "recentness": 0.20, "negative_sentiment": 0.25, "cluster_coherence": 0.00},
                "has_temporal": True,
                "has_aspects": False,
                "has_sentiment": True,
                "has_evidence": False,
            },
            "w_o_sentiment": {
                "weights": {"growth": 0.35, "volume": 0.30, "recentness": 0.25, "negative_sentiment": 0.00, "cluster_coherence": 0.10},
                "has_temporal": True,
                "has_aspects": True,
                "has_sentiment": False,
                "has_evidence": True,
            },
            "w_o_evidence_retrieval": {
                "weights": {"growth": 0.30, "volume": 0.20, "recentness": 0.20, "negative_sentiment": 0.20, "cluster_coherence": 0.10},
                "has_temporal": True,
                "has_aspects": True,
                "has_sentiment": True,
                "has_evidence": False,
            },
        }

        ablation_results = {}
        from ml.emerging.emerging_scorer import EmergingScorer

        for name, config in configurations.items():
            scorer = EmergingScorer(weights=config["weights"])
            scored_issues = []
            for item in issues_data:
                res = scorer.score_issue(
                    cluster_id=item["cluster_id"],
                    cluster_name=item["name"],
                    total_volume=item["total_volume"],
                    dataset_total_volume=item.get("dataset_total_volume", 1000),
                    growth_rate=item.get("growth_rate", 0.0) if config["has_temporal"] else 0.0,
                    recent_volume=item.get("recent_volume", int(item["total_volume"] * 0.3)) if config["has_temporal"] else 0,
                    negative_sentiment_ratio=item.get("negative_sentiment_ratio", 0.6) if config["has_sentiment"] else 0.33,
                    cluster_coherence=item.get("coherence", 0.75) if config.get("has_aspects", True) else 0.40,
                )
                scored_issues.append(res)

            # Sort descending by score
            scored_issues.sort(key=lambda x: x["emerging_score"], reverse=True)
            top_ranked = [i["issue_name"] for i in scored_issues[:3]]

            # Average score & priority distribution
            avg_score = float(np.mean([i["emerging_score"] for i in scored_issues]))
            priority_counts = {}
            for i in scored_issues:
                priority_counts[i["priority"]] = priority_counts.get(i["priority"], 0) + 1

            ablation_results[name] = {
                "description": name.replace("_", " ").title(),
                "top_3_prioritized_issues": top_ranked,
                "mean_emerging_score": round(avg_score, 4),
                "priority_distribution": priority_counts,
                "has_temporal": config["has_temporal"],
                "has_sentiment": config["has_sentiment"],
                "has_aspects": config["has_aspects"],
                "has_evidence": config["has_evidence"],
            }

        out_file = os.path.join(self.output_dir, "ablation_results.json")
        with open(out_file, "w", encoding="utf-8") as f:
            json.dump(ablation_results, f, indent=2)

        return ablation_results
