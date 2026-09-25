import re
import html
import unicodedata
from typing import Dict, List, Optional, Tuple


class TextPreprocessor:
    """
    Configurable text preprocessing module for multi-source customer text.
    Maintains dual representations:
    1. clean_text: preserves casing, grammar, and emotional cues for transformer models.
    2. normalized_text: lowercased, token-normalized for lexical indexing & baselines.
    """

    def __init__(
        self,
        min_words: int = 2,
        max_words: int = 1000,
        normalize_repetition: bool = True,
        preserve_emojis: bool = True,
    ):
        self.min_words = min_words
        self.max_words = max_words
        self.normalize_repetition = normalize_repetition
        self.preserve_emojis = preserve_emojis

        # Precompiled regular expressions
        self._html_tag_re = re.compile(r"<[^>]+>")
        self._url_re = re.compile(r"https?://\S+|www\.\S+")
        self._whitespace_re = re.compile(r"\s+")
        self._user_mention_re = re.compile(r"@[\w_]+")
        self._repeated_char_re = re.compile(r"(.)\1{2,}")

    def clean(self, raw_text: Optional[str]) -> str:
        """
        Cleans text for Transformer models. Preserves punctuation, sentiment cues,
        and casing while removing HTML, resolving entities, and standardizing whitespace.
        """
        if not raw_text or not isinstance(raw_text, str):
            return ""

        # Decode HTML entities
        text = html.unescape(raw_text)

        # Remove HTML tags
        text = self._html_tag_re.sub(" ", text)

        # Normalize Unicode characters
        text = unicodedata.normalize("NFKC", text)

        # Normalize URLs
        text = self._url_re.sub(" ", text)

        # Normalize excessive repeated characters (e.g., 'sooooo' -> 'soo')
        if self.normalize_repetition:
            text = self._repeated_char_re.sub(r"\1\1", text)

        # Normalize whitespace
        text = self._whitespace_re.sub(" ", text).strip()

        return text

    def normalize_for_lexical(self, text: str) -> str:
        """
        Produces a lowercased, simplified representation suited for TF-IDF
        and keyword-based baseline comparisons.
        """
        cleaned = self.clean(text).lower()
        # Remove mentions
        cleaned = self._user_mention_re.sub(" ", cleaned)
        # Strip non-alphanumeric except basic spacing
        cleaned = re.sub(r"[^\w\s]", " ", cleaned)
        return self._whitespace_re.sub(" ", cleaned).strip()

    def detect_language(self, text: str) -> str:
        """
        Heuristic language detection. Detects English vs Non-English based on
        ASCII ratio and basic lexical features.
        """
        if not text:
            return "unknown"
        ascii_chars = sum(1 for c in text if ord(c) < 128)
        ratio = ascii_chars / len(text)
        if ratio > 0.85:
            return "en"
        return "non-en"

    def is_valid(self, text: str) -> Tuple[bool, Optional[str]]:
        """
        Validates text against quality requirements.
        Returns (is_valid, rejection_reason).
        """
        if not text or not text.strip():
            return False, "Empty text"
        words = text.strip().split()
        if len(words) < self.min_words:
            return False, f"Too short (less than {self.min_words} words)"
        if len(words) > self.max_words:
            return False, f"Too long (greater than {self.max_words} words)"
        return True, None

    def process_record(self, raw_text: str) -> Dict[str, any]:
        """
        Full preprocessing pass for a single text record.
        """
        clean_text = self.clean(raw_text)
        is_valid, reason = self.is_valid(clean_text)
        normalized = self.normalize_for_lexical(clean_text) if is_valid else ""
        lang = self.detect_language(clean_text) if is_valid else "unknown"

        return {
            "original_text": raw_text,
            "clean_text": clean_text,
            "normalized_text": normalized,
            "language": lang,
            "word_count": len(clean_text.split()) if is_valid else 0,
            "is_valid": is_valid,
            "rejection_reason": reason,
        }
