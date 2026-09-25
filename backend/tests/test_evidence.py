import pytest
import numpy as np
from ml.evidence.evidence_retriever import EvidenceRetriever


def test_evidence_retriever_partitioning():
    er = EvidenceRetriever()
    # Mock embeddings
    dim = 384
    embeddings = np.random.randn(5, dim)
    embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

    texts = [
        "Payment broke after update, error 500",
        "Payment is completely failed, money lost",
        "Payment was quick and smooth, loved it",
        "Checkout was instantaneous and easy",
        "What are the payment options?",
    ]
    sentiments = ["negative", "negative", "positive", "positive", "neutral"]

    res = er.retrieve_evidence(
        query="payment failed",
        texts=texts,
        embeddings=embeddings,
        sentiments=sentiments,
        similarity_threshold=-1.0,  # capture all for test
    )

    assert len(res["supporting"]) > 0
    assert len(res["contradicting"]) > 0
    assert "evidence_balance_ratio" in res["stats"]
