import re
import logging
from typing import Dict, List, Optional, Set, Tuple
import numpy as np

logger = logging.getLogger(__name__)


class AspectExtractor:
    """
    Hybrid semantic aspect extraction engine.
    1. Extracts candidate noun-phrases, frequent terms, and contextual patterns.
    2. Maps candidate phrases via SentenceEmbeddings to canonical semantic aspects.
    3. Discovers dynamic open-domain aspects when phrases fall outside standard clusters.
    """

    DEFAULT_CANONICAL_ASPECTS = {
        "PAYMENT": "payment transaction upi billing credit card checkout failure charge money fee",
        "CUSTOMER_SERVICE": "customer service agent representative support staff phone call center help desk",
        "FLIGHT_OPERATIONS": "flight delay cancellation late departure arrival schedule missed connection gate runway pilot",
        "BAGGAGE": "baggage luggage suitcase lost bag claim carousel damaged belongings",
        "BOOKING_TICKETING": "booking ticket reservation seat assignment confirmation rebook itinerary cancel ticket",
        "CABIN_COMFORT": "seat legroom space food meal drink wifi tv entertainment cleanliness comfort temperature",
        "DIGITAL_APP": "mobile app website online portal login crash update bug interface error password",
        "PRICING_REFUND": "price cost expensive fare refund compensation voucher overcharge policy reimbursement",
    }

    # Stopwords to filter out trivial candidate phrases
    STOPWORDS = {
        "a", "an", "the", "and", "or", "but", "if", "because", "as", "what",
        "which", "this", "that", "these", "those", "then", "just", "so", "than",
        "such", "both", "through", "about", "for", "is", "of", "while", "during",
        "to", "from", "in", "out", "on", "off", "again", "further", "then", "once",
        "here", "there", "when", "where", "why", "how", "all", "any", "both",
        "each", "few", "more", "most", "other", "some", "no", "nor", "not", "only",
        "own", "same", "very", "can", "will", "don", "should", "now", "i", "me",
        "my", "we", "our", "you", "your", "they", "them", "he", "she", "it",
        "am", "are", "was", "were", "be", "been", "being", "have", "has", "had",
        "do", "does", "did", "flight", "plane", "airline", "get", "got", "go",
    }

    def __init__(
        self,
        embedder=None,
        similarity_threshold: float = 0.45,
        custom_canonical_aspects: Optional[Dict[str, str]] = None,
    ):
        self.similarity_threshold = similarity_threshold
        self._embedder = embedder
        self.canonical_aspects = dict(self.DEFAULT_CANONICAL_ASPECTS)
        if custom_canonical_aspects:
            self.canonical_aspects.update(custom_canonical_aspects)

        self._aspect_names: List[str] = list(self.canonical_aspects.keys())
        self._aspect_vectors: Optional[np.ndarray] = None
        self._discovered_aspects: Dict[str, np.ndarray] = {}

    def _ensure_vectors(self):
        if self._aspect_vectors is None:
            if self._embedder is None:
                from ml.embeddings.sentence_embedder import SentenceEmbedder
                self._embedder = SentenceEmbedder()

            descriptions = [self.canonical_aspects[a] for a in self._aspect_names]
            self._aspect_vectors = self._embedder.encode(descriptions, use_cache=True)

    def extract_candidates(self, text: str) -> List[str]:
        """
        Extract candidate aspect phrases (unigrams and bigrams/trigrams)
        using regex pattern matching and stopword filtering.
        """
        if not text:
            return []

        clean = re.sub(r"[^\w\s-]", " ", text.lower())
        tokens = [w for w in clean.split() if len(w) > 2]

        candidates: List[str] = []

        # Extract meaningful unigrams
        for word in tokens:
            if word not in self.STOPWORDS and not word.isdigit():
                candidates.append(word)

        # Extract bigrams
        for i in range(len(tokens) - 1):
            w1, w2 = tokens[i], tokens[i + 1]
            if (w1 not in self.STOPWORDS or w2 not in self.STOPWORDS) and not (w1.isdigit() and w2.isdigit()):
                if len(w1) > 2 and len(w2) > 2:
                    candidates.append(f"{w1} {w2}")

        # Deduplicate while preserving order
        seen = set()
        deduped = []
        for c in candidates:
            if c not in seen:
                seen.add(c)
                deduped.append(c)

        return deduped[:12]

    def extract_aspects(self, text: str) -> List[Dict[str, any]]:
        """
        Extract and normalize aspects from a given text.
        Returns list of dicts:
        [
            {"aspect": "PAYMENT", "matched_term": "credit card", "confidence": 0.78, "type": "canonical"},
            ...
        ]
        """
        candidates = self.extract_candidates(text)
        if not candidates:
            return []

        self._ensure_vectors()

        candidate_vectors = self._embedder.encode(candidates, use_cache=True)
        # Compute cosine similarity between candidates and canonical aspects
        # Shape: (num_candidates, num_aspects)
        sim_matrix = np.dot(candidate_vectors, self._aspect_vectors.T)

        extracted = []
        detected_aspects = set()

        for c_idx, candidate in enumerate(candidates):
            best_aspect_idx = int(np.argmax(sim_matrix[c_idx]))
            max_sim = float(sim_matrix[c_idx, best_aspect_idx])

            if max_sim >= self.similarity_threshold:
                aspect_name = self._aspect_names[best_aspect_idx]
                if aspect_name not in detected_aspects:
                    detected_aspects.add(aspect_name)
                    extracted.append({
                        "aspect": aspect_name,
                        "matched_term": candidate,
                        "confidence": round(max_sim, 4),
                        "type": "canonical",
                    })
            elif max_sim < 0.35 and len(candidate.split()) > 1:
                # Potential discovered domain aspect
                discovered_name = candidate.upper().replace(" ", "_")
                if discovered_name not in detected_aspects and len(detected_aspects) < 4:
                    detected_aspects.add(discovered_name)
                    extracted.append({
                        "aspect": discovered_name,
                        "matched_term": candidate,
                        "confidence": round(0.50, 4),
                        "type": "discovered",
                    })

        # Sort by confidence descending
        extracted.sort(key=lambda x: x["confidence"], reverse=True)
        return extracted[:5]
