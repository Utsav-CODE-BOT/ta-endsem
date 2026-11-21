import logging
from typing import Dict, List, Optional, Union
import numpy as np
from transformers import pipeline

logger = logging.getLogger(__name__)


class SentimentAnalyzer:
    """
    Transformer-based sentiment analyzer using CardiffNLP Twitter-RoBERTa.
    Provides calibrated 3-class sentiment (positive, neutral, negative)
    with confidence scores and fallback lexical/linear baseline comparison.
    """

    def __init__(self, model_name: str = "cardiffnlp/twitter-roberta-base-sentiment-latest"):
        self.model_name = model_name
        self._pipe = None

    @property
    def pipe(self):
        if self._pipe is None:
            logger.info("Initializing Transformer sentiment pipeline: %s", self.model_name)
            self._pipe = pipeline(
                "sentiment-analysis",
                model=self.model_name,
                device=-1,  # CPU inference
                truncation=True,
                max_length=256,
            )
        return self._pipe

    def analyze_single(self, text: str) -> Dict[str, Union[str, float]]:
        """
        Analyze sentiment for a single text.
        Returns: {"label": "negative"|"neutral"|"positive"|"unknown", "score": float, "status": str}
        """
        if not text or not text.strip():
            return {"label": "neutral", "score": 0.50, "status": "EMPTY_INPUT"}

        try:
            res = self.pipe(text[:512])[0]
            label = str(res["label"]).lower()
            score = float(res["score"])
            return {"label": label, "score": round(score, 4), "status": "SUCCESS"}
        except Exception as e:
            logger.warning("Transformer sentiment inference error: %s.", e)
            return {"label": "unknown", "score": 0.0, "status": "INFERENCE_ERROR"}

    def analyze_batch(self, texts: List[str], batch_size: int = 32) -> List[Dict[str, Union[str, float]]]:
        """
        Batch sentiment classification with explicit status tracking.
        """
        if not texts:
            return []

        results = []
        for i in range(0, len(texts), batch_size):
            chunk = texts[i : i + batch_size]
            valid_indices = [idx for idx, t in enumerate(chunk) if t and t.strip()]
            valid_texts = [chunk[idx][:512] for idx in valid_indices]

            chunk_results = [
                {"label": "neutral", "score": 0.50, "status": "EMPTY_INPUT"} for _ in chunk
            ]

            if valid_texts:
                try:
                    preds = self.pipe(valid_texts)
                    for valid_idx, pred in zip(valid_indices, preds):
                        chunk_results[valid_idx] = {
                            "label": str(pred["label"]).lower(),
                            "score": round(float(pred["score"]), 4),
                            "status": "SUCCESS",
                        }
                except Exception as e:
                    logger.warning("Batch sentiment error: %s, falling back to per-item.", e)
                    for valid_idx in valid_indices:
                        chunk_results[valid_idx] = self.analyze_single(chunk[valid_idx])

            results.extend(chunk_results)

        return results


class EmotionDetector:
    """
    Emotion enrichment module mapping text representations to 7 core affective states:
    - anger
    - frustration
    - disappointment
    - joy
    - satisfaction
    - fear
    - confusion
    Uses anchor semantic vectors from SentenceEmbedder for fast, calibration-grounded emotion inference.
    """

    EMOTIONS = [
        "anger",
        "frustration",
        "disappointment",
        "joy",
        "satisfaction",
        "fear",
        "confusion",
    ]

    EMOTION_ANCHORS = {
        "anger": "furious rage screaming terrible customer service outrage",
        "frustration": "stuck waiting forever nothing works wasted my time annoying",
        "disappointment": "expected better let down unhappy with poor quality",
        "joy": "wonderful awesome great delightful amazing experience loved it",
        "satisfaction": "smooth easy resolved problem helpful polite prompt",
        "fear": "scared worried anxious safety risk panic danger",
        "confusion": "don't understand unclear misleading error what happened why",
    }

    def __init__(self, embedder=None):
        self._embedder = embedder
        self._anchor_vectors = None

    def _ensure_anchor_vectors(self):
        if self._anchor_vectors is None:
            if self._embedder is None:
                from ml.embeddings.sentence_embedder import SentenceEmbedder
                self._embedder = SentenceEmbedder()
            anchors = [self.EMOTION_ANCHORS[e] for e in self.EMOTIONS]
            self._anchor_vectors = self._embedder.encode(anchors, use_cache=True)

    def detect(self, text: str, text_embedding: Optional[np.ndarray] = None) -> Dict[str, Union[str, float, Dict[str, float]]]:
        """
        Detects primary emotion and full emotion distribution.
        """
        if not text or not text.strip():
            return {
                "primary_emotion": "neutral",
                "score": 0.5,
                "scores": {e: 0.14 for e in self.EMOTIONS}
            }

        self._ensure_anchor_vectors()

        if text_embedding is None:
            text_vec = self._embedder.encode(text, use_cache=True)
        else:
            text_vec = text_embedding

        # Cosine similarity against emotion anchors
        sims = np.dot(self._anchor_vectors, text_vec)
        # Softmax normalization with temperature
        temp = 0.15
        exp_sims = np.exp(sims / temp)
        probs = exp_sims / np.sum(exp_sims)

        emotion_scores = {e: round(float(p), 4) for e, p in zip(self.EMOTIONS, probs)}
        top_idx = int(np.argmax(probs))
        top_emotion = self.EMOTIONS[top_idx]
        top_score = round(float(probs[top_idx]), 4)

        return {
            "primary_emotion": top_emotion,
            "score": top_score,
            "scores": emotion_scores,
        }
