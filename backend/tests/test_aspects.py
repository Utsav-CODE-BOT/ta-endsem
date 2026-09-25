import pytest
from ml.aspects.aspect_extractor import AspectExtractor


def test_candidate_extraction():
    ae = AspectExtractor()
    text = "The payment failed at checkout and credit card was charged twice"
    candidates = ae.extract_candidates(text)
    assert any("payment" in c or "checkout" in c or "credit card" in c for c in candidates)


def test_aspect_normalization():
    ae = AspectExtractor()
    text = "Baggage claim carousel was stuck and my suitcase was missing"
    aspects = ae.extract_aspects(text)
    assert len(aspects) > 0
    aspect_names = [a["aspect"] for a in aspects]
    assert "BAGGAGE" in aspect_names
