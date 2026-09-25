import pytest
from ml.preprocessing.text_cleaner import TextPreprocessor


def test_text_cleaner_html_and_unicode():
    tp = TextPreprocessor()
    raw = "<p>Flight was sooooo bad &amp; delayed! https://example.com</p>"
    clean = tp.clean(raw)
    assert "<p>" not in clean
    assert "&amp;" not in clean
    assert "https://" not in clean
    assert "soo bad" in clean or "bad" in clean


def test_text_normalization_for_lexical():
    tp = TextPreprocessor()
    raw = "@airline Why is baggage claim ALWAYS broken?!"
    norm = tp.normalize_for_lexical(raw)
    assert "@airline" not in norm
    assert "?" not in norm
    assert "baggage" in norm
    assert "broken" in norm


def test_validation_and_language():
    tp = TextPreprocessor(min_words=2)
    assert tp.is_valid("")[0] is False
    assert tp.is_valid("hello")[0] is False
    assert tp.is_valid("Flight was great and comfortable")[0] is True
    assert tp.detect_language("Customer support agent resolved my issue promptly") == "en"
