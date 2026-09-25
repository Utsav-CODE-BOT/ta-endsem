import logging
from typing import Dict, List, Optional
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class EvidenceRetriever:
    """
    Evidence retrieval engine supporting both supporting and contradictory evidence discovery.
    Prevents confirmation bias and hallucinated root causes by:
    1. Semantically retrieving customer statements mentioning the factor or issue.
    2. Segmenting evidence by sentiment valence:
       - Supporting: Confirms failure, disruption, or complaint.
       - Contradicting: Positive customer experience or smooth operation despite factor presence.
       - Neutral: Factual inquiry or neutral operational report.
    3. Computing calibrated evidence balance statistics.
    """

    def __init__(self, embedder=None, top_k: int = 15):
        self._embedder = embedder
        self.top_k = top_k

    def retrieve_evidence(
        self,
        query: str,
        texts: List[str],
        embeddings: np.ndarray,
        sentiments: List[str],
        dates: Optional[List[str]] = None,
        sources: Optional[List[str]] = None,
        similarity_threshold: float = 0.35,
    ) -> Dict[str, any]:
        """
        Retrieves top supporting, contradictory, and neutral evidence for a hypothesis.
        """
        if not texts or len(embeddings) == 0:
            return {
                "query": query,
                "supporting": [],
                "contradicting": [],
                "neutral": [],
                "stats": {
                    "total_retrieved": 0,
                    "supporting_count": 0,
                    "contradicting_count": 0,
                    "neutral_count": 0,
                    "evidence_balance_ratio": 1.0,
                },
            }

        if self._embedder is None:
            from ml.embeddings.sentence_embedder import SentenceEmbedder
            self._embedder = SentenceEmbedder()

        query_vec = self._embedder.encode(query, use_cache=True)
        # Cosine similarity across all embeddings
        sims = np.dot(embeddings, query_vec).flatten()

        # Sort indices by similarity descending
        ranked_indices = sims.argsort()[::-1]

        supporting = []
        contradicting = []
        neutral = []

        total_qualifying = 0

        for idx in ranked_indices:
            score = float(sims[idx])
            if score < similarity_threshold:
                break

            total_qualifying += 1
            sent = sentiments[idx] if idx < len(sentiments) else "neutral"
            date_str = dates[idx] if dates and idx < len(dates) else "N/A"
            source_str = sources[idx] if sources and idx < len(sources) else "Customer Feedback"

            evidence_item = {
                "index": int(idx),
                "text": texts[idx],
                "similarity": round(score, 4),
                "sentiment": sent,
                "date": date_str,
                "source": source_str,
            }

            if sent == "negative":
                if len(supporting) < self.top_k:
                    supporting.append(evidence_item)
            elif sent == "positive":
                if len(contradicting) < self.top_k:
                    contradicting.append(evidence_item)
            else:
                if len(neutral) < self.top_k:
                    neutral.append(evidence_item)

        supp_count = len(supporting)
        contra_count = len(contradicting)
        neut_count = len(neutral)
        balance_ratio = round(supp_count / max(1, contra_count), 2)

        return {
            "query": query,
            "supporting": supporting,
            "contradicting": contradicting,
            "neutral": neutral,
            "stats": {
                "total_qualifying_statements": total_qualifying,
                "supporting_count": supp_count,
                "contradicting_count": contra_count,
                "neutral_count": neut_count,
                "evidence_balance_ratio": balance_ratio,
            },
        }
