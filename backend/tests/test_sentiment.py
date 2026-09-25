import pytest
from ml.sentiment.sentiment_analyzer import SentimentAnalyzer, EmotionDetector


def test_sentiment_analyzer():
    sa = SentimentAnalyzer()
    res_neg = sa.analyze_single("Terrible flight, lost my luggage and delayed for 8 hours!")
    assert res_neg["label"] in ["negative", "neutral", "positive"]
    assert "score" in res_neg
    assert isinstance(res_neg["score"], float)

    res_pos = sa.analyze_single("Loved the smooth flight and amazing crew, wonderful service!")
    assert res_pos["label"] in ["positive", "neutral"]


def test_emotion_detector():
    ed = EmotionDetector()
    res = ed.detect("I am so furious and angry with this terrible customer service!")
    assert res["primary_emotion"] in EmotionDetector.EMOTIONS
    assert res["score"] > 0
    assert len(res["scores"]) == 7
