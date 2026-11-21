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


def test_hypothesis_aware_stance_classification():
    er = EvidenceRetriever()
    factor = "Mobile App v4.2"

    # Example 1: Factor present + complaint -> SUPPORTING
    s1 = er._determine_stance("After v4.2 the payment checkout fails every time.", factor, "negative")
    assert s1 == "SUPPORTING"

    # Example 2: Factor present + positive resolution -> CONTRADICTING
    s2 = er._determine_stance("Checkout works normally on v4.2.", factor, "positive")
    assert s2 == "CONTRADICTING"

    # Example 3: Factor absent -> UNCERTAIN (cannot be factor-specific evidence even if positive)
    s3 = er._determine_stance("Customer service was helpful.", factor, "positive")
    assert s3 == "UNCERTAIN"

    # Example 4: Mixed cues -> UNCERTAIN (do not force contradicting because of positive words)
    s4 = er._determine_stance("Excellent update, another payment failure!", factor, "negative")
    assert s4 == "UNCERTAIN"

    # Example 5: Factor absent + negative -> UNCERTAIN (cannot be factor-specific supporting)
    s5 = er._determine_stance("The flight was delayed.", factor, "negative")
    assert s5 == "UNCERTAIN"
