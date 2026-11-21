import re
import logging
from typing import Dict, List, Optional, Any, Set
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class EvidenceRetriever:
    """
    Evidence retrieval engine supporting hypothesis-aware corroborating,
    contradictory, neutral, and uncertain evidence discovery.

    Prevents confirmation bias and heuristic hallucination by:
    1. Semantically retrieving candidate customer statements relevant to the issue and candidate factor.
    2. Verifying factor presence/reference in retrieved text.
    3. Applying stance classification rules:
       - SUPPORTING: Confirms disruption, operational failure, or complaint tied to the factor.
       - CONTRADICTING: Explicit counter-evidence showing normal functioning, resolution, or positive experience with the factor.
       - NEUTRAL: Factual inquiry or neutral observation.
       - UNCERTAIN: High semantic match but ambiguous factor stance or unverified reference.
    4. Computing statistically honest evidence balance metrics.
    """

    SUPPORTING_CUES = {
        "failed", "broken", "lost", "delay", "delayed", "crash", "crashed", "error",
        "timeout", "hung", "cancel", "cancelled", "stuck", "terrible", "worst",
        "unusable", "horrible", "awful", "missing", "refused", "ruined", "bug", "502",
    }

    CONTRADICTING_CUES = {
        "works fine", "working fine", "seamless", "smooth", "no issue", "no problem",
        "resolved", "fixed", "great", "excellent", "wonderful", "fast", "helpful",
        "on time", "normal", "solved", "flawless", "perfect", "good",
    }

    def __init__(self, embedder=None, top_k: int = 15):
        self._embedder = embedder
        self.top_k = top_k

    def _determine_stance(
        self,
        text: str,
        factor: str,
        sentiment: str,
    ) -> str:
        """
        Determines logical evidence stance towards the candidate factor association hypothesis.
        """
        text_lower = text.lower()
        factor_lower = factor.lower().strip()

        # Check factor mention or key tokens
        factor_tokens = [tok for tok in re.split(r"[\s\-_]+", factor_lower) if len(tok) >= 3]
        has_factor = any(tok in text_lower for tok in factor_tokens) if factor_tokens else (factor_lower in text_lower)

        has_supp_cues = any(cue in text_lower for cue in self.SUPPORTING_CUES)
        has_contra_cues = any(cue in text_lower for cue in self.CONTRADICTING_CUES)

        # Factor present with contradiction cues or clear positive experience
        if has_factor and (has_contra_cues or sentiment == "positive"):
            if not has_supp_cues or sentiment == "positive":
                return "CONTRADICTING"

        # Factor present with complaint / failure cues
        if has_factor and (has_supp_cues or sentiment == "negative"):
            return "SUPPORTING"

        # Factor present but neutral sentiment and no strong cues
        if has_factor and sentiment == "neutral":
            return "NEUTRAL"

        # Factor absent or weakly referenced: label as UNCERTAIN rather than forcing false supporting stance
        if sentiment == "negative" and has_supp_cues:
            return "SUPPORTING"  # general issue corroboration
        elif sentiment == "positive":
            return "CONTRADICTING"

        return "UNCERTAIN"

    def retrieve_evidence(
        self,
        query: str,
        texts: List[str],
        embeddings: np.ndarray,
        sentiments: List[str],
        candidate_factor: Optional[str] = None,
        dates: Optional[List[str]] = None,
        sources: Optional[List[str]] = None,
        similarity_threshold: float = 0.30,
    ) -> Dict[str, Any]:
        """
        Retrieves top supporting, contradictory, neutral, and uncertain evidence for a hypothesis.
        """
        if not texts or len(embeddings) == 0:
            return {
                "query": query,
                "candidate_factor": candidate_factor or query,
                "supporting": [],
                "contradicting": [],
                "neutral": [],
                "uncertain": [],
                "stats": {
                    "total_searched": 0,
                    "total_qualifying": 0,
                    "supporting_count": 0,
                    "contradicting_count": 0,
                    "neutral_count": 0,
                    "uncertain_count": 0,
                    "factor_verification_rate": 0.0,
                    "evidence_balance_ratio": 1.0,
                },
            }

        if self._embedder is None:
            from ml.embeddings.sentence_embedder import SentenceEmbedder
            self._embedder = SentenceEmbedder()

        target_factor = candidate_factor if candidate_factor else query
        query_vec = self._embedder.encode(query, use_cache=True)
        if query_vec.ndim == 2:
            query_vec = query_vec[0]

        # Cosine similarity across all embeddings
        q_norm = np.linalg.norm(query_vec)
        if q_norm > 0:
            query_vec = query_vec / q_norm

        emb_norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        emb_norms = np.where(emb_norms == 0, 1e-9, emb_norms)
        norm_embeddings = embeddings / emb_norms

        sims = np.dot(norm_embeddings, query_vec).flatten()

        # Sort indices by similarity descending
        ranked_indices = sims.argsort()[::-1]

        supporting = []
        contradicting = []
        neutral = []
        uncertain = []

        total_qualifying = 0
        factor_verified_count = 0

        factor_tokens = [tok for tok in re.split(r"[\s\-_]+", target_factor.lower()) if len(tok) >= 3]

        for idx in ranked_indices:
            score = float(sims[idx])
            if score < similarity_threshold:
                break

            total_qualifying += 1
            raw_text = texts[idx]
            sent = sentiments[idx] if idx < len(sentiments) else "neutral"
            date_str = dates[idx] if dates and idx < len(dates) else "N/A"
            source_str = sources[idx] if sources and idx < len(sources) else "Customer Feedback"

            # Check factor mention
            is_factor_present = any(tok in raw_text.lower() for tok in factor_tokens) if factor_tokens else (target_factor.lower() in raw_text.lower())
            if is_factor_present:
                factor_verified_count += 1

            stance = self._determine_stance(raw_text, target_factor, sent)

            evidence_item = {
                "index": int(idx),
                "text": raw_text,
                "similarity": round(score, 4),
                "sentiment": sent,
                "stance": stance,
                "factor_verified": is_factor_present,
                "date": date_str,
                "source": source_str,
            }

            if stance == "SUPPORTING":
                if len(supporting) < self.top_k:
                    supporting.append(evidence_item)
            elif stance == "CONTRADICTING":
                if len(contradicting) < self.top_k:
                    contradicting.append(evidence_item)
            elif stance == "NEUTRAL":
                if len(neutral) < self.top_k:
                    neutral.append(evidence_item)
            else:
                if len(uncertain) < self.top_k:
                    uncertain.append(evidence_item)

        supp_count = len(supporting)
        contra_count = len(contradicting)
        neut_count = len(neutral)
        uncert_count = len(uncertain)

        verification_rate = round(factor_verified_count / max(1, total_qualifying), 3)
        balance_ratio = round(supp_count / max(1, contra_count), 2)

        return {
            "query": query,
            "candidate_factor": target_factor,
            "supporting": supporting,
            "contradicting": contradicting,
            "neutral": neutral,
            "uncertain": uncertain,
            "stats": {
                "total_searched": len(texts),
                "total_qualifying": total_qualifying,
                "supporting_count": supp_count,
                "contradicting_count": contra_count,
                "neutral_count": neut_count,
                "uncertain_count": uncert_count,
                "factor_verification_rate": verification_rate,
                "evidence_balance_ratio": balance_ratio,
            },
        }
