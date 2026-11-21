"""
Deterministic Sample Selection Utility for VoiceMiner Research Benchmarks.

Provides a unified, auditable selection mechanism ensuring that BenchmarkRunner (RQ1-RQ3)
and AblationStudyRunner (RQ4-RQ5) evaluate the exact same population of preprocessed customer
records with preserved identifiers, timestamps, sentiment labels, and metadata.
"""

from dataclasses import dataclass
from typing import Dict, List, Any, Optional
import pandas as pd
from ml.preprocessing.text_cleaner import TextPreprocessor


@dataclass
class EvaluationSample:
    """Represents a unified, verified evaluation population."""
    dataset_name: str
    raw_sample_size: int
    post_filtering_sample_size: int
    record_ids: List[Any]
    raw_texts: List[str]
    clean_texts: List[str]
    timestamps: List[Any]
    sentiments: List[str]
    metadata: List[Dict[str, Any]]
    df: pd.DataFrame
    random_seed: Optional[int] = 42

    def to_dict(self) -> Dict[str, Any]:
        return {
            "dataset_name": self.dataset_name,
            "raw_sample_size": self.raw_sample_size,
            "post_filtering_sample_size": self.post_filtering_sample_size,
            "record_ids": self.record_ids,
            "random_seed": self.random_seed,
        }


class DeterministicSampleSelector:
    """
    Deterministic evaluation sample selector.
    Extracts the first N records (or deterministic sample), cleans texts with TextPreprocessor,
    filters invalid/short items consistently, and retains 1-to-1 alignment between
    record IDs, clean texts, parsed timestamps, sentiments, and metadata.
    """

    def __init__(self, min_word_count: int = 3):
        self.min_word_count = min_word_count
        self.preprocessor = TextPreprocessor()

    def select_sample(
        self,
        df: pd.DataFrame,
        sample_size: int = 500,
        random_seed: int = 42,
        dataset_name: str = "airline_sample_1000.csv",
    ) -> EvaluationSample:
        # Determine candidate columns
        text_col = None
        for col in ["text", "clean_text", "comment", "feedback"]:
            if col in df.columns:
                text_col = col
                break
        if not text_col:
            raise ValueError(f"No text column found in dataframe. Columns: {list(df.columns)}")

        time_col = None
        for col in ["tweet_created", "timestamp", "date", "created_at"]:
            if col in df.columns:
                time_col = col
                break

        sent_col = None
        for col in ["airline_sentiment", "sentiment"]:
            if col in df.columns:
                sent_col = col
                break

        id_col = None
        for col in ["tweet_id", "id", "record_id"]:
            if col in df.columns:
                id_col = col
                break

        # Select window of input records deterministically (first sample_size rows)
        raw_slice = df.iloc[:sample_size].copy()
        raw_count = len(raw_slice)

        valid_records = []
        for orig_idx, row in raw_slice.iterrows():
            raw_text = row.get(text_col)
            if pd.isna(raw_text):
                continue
            cleaned = self.preprocessor.clean(str(raw_text))
            tokens = cleaned.split()
            if len(tokens) < self.min_word_count:
                continue

            rec_id = row.get(id_col) if id_col and pd.notna(row.get(id_col)) else int(orig_idx)
            raw_t = row.get(time_col) if time_col else None
            parsed_t = pd.to_datetime(raw_t, utc=True) if pd.notna(raw_t) else None
            sent = str(row.get(sent_col)).lower() if (sent_col and pd.notna(row.get(sent_col))) else "neutral"

            # Metadata extraction without reference label leakage
            meta = {}
            for col in raw_slice.columns:
                if col not in [text_col, time_col, sent_col, id_col] and pd.notna(row[col]):
                    meta[col] = row[col]

            valid_records.append({
                "record_id": rec_id,
                "raw_text": str(raw_text),
                "clean_text": cleaned,
                "timestamp": parsed_t,
                "sentiment": sent,
                "metadata": meta,
                "original_index": int(orig_idx),
            })

        valid_df = pd.DataFrame(valid_records)
        post_count = len(valid_records)

        return EvaluationSample(
            dataset_name=dataset_name,
            raw_sample_size=raw_count,
            post_filtering_sample_size=post_count,
            record_ids=[r["record_id"] for r in valid_records],
            raw_texts=[r["raw_text"] for r in valid_records],
            clean_texts=[r["clean_text"] for r in valid_records],
            timestamps=[r["timestamp"] for r in valid_records],
            sentiments=[r["sentiment"] for r in valid_records],
            metadata=[r["metadata"] for r in valid_records],
            df=valid_df,
            random_seed=random_seed,
        )
