"""
Tests for Atomic Database Rollback, Snapshot Concurrency Consistency,
and Honest Missing-Timestamp Fallbacks.
"""

import threading
import time
import pytest
import pandas as pd
from unittest.mock import MagicMock
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from backend.app.database.session import Base
from backend.app.models.database import IssueModel, DocumentModel
from backend.app.services.pipeline_service import PipelineService, AnalysisSnapshot


@pytest.fixture
def in_memory_db():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(bind=engine)
    Session = sessionmaker(bind=engine)
    session = Session()
    yield session
    session.close()


def test_database_atomic_rollback_preserves_previous_data(in_memory_db):
    service = PipelineService()

    # Pre-seed database with an existing initial record
    prior_doc = DocumentModel(
        id="initial_doc_1",
        source="Customer Review",
        original_text="Previous existing feedback.",
        clean_text="Previous existing feedback.",
        language="en",
        word_count=3,
    )
    in_memory_db.add(prior_doc)
    in_memory_db.commit()

    # Assert pre-seeded record exists
    assert in_memory_db.query(DocumentModel).count() == 1

    # Attempt saving faulty data that triggers an insertion error (e.g. malformed null constraints)
    faulty_df = pd.DataFrame([{
        "id": "new_doc_2",
        "source": "Twitter",
        "original_text": "Flight was cancelled",
        "clean_text": "Flight was cancelled",
        "timestamp": None,
        "language": "en",
        "word_count": 3,
        "sentiment": "negative",
        "sentiment_score": 0.95,
        "emotion": "anger",
        "emotion_score": 0.88,
        "aspects": [],
        "cluster_id": "NOT_AN_INTEGER_RAISES_ERROR",  # Triggers database error in IssueMembershipModel
        "coord_x": 0.0,
        "coord_y": 0.0,
    }])

    clusters = [{"cluster_id": 0, "name": "Flight Delays", "size": 1, "percentage": 100.0, "top_terms": ["flight"], "center_2d": [0.0, 0.0]}]
    emerging = []
    hypotheses = []

    success, err = service._save_to_database(
        db=in_memory_db,
        df=faulty_df,
        clusters=clusters,
        emerging=emerging,
        hypotheses=hypotheses,
    )

    # Must fail cleanly and report error
    assert success is False
    assert err is not None

    # Rollback must preserve previous data completely (count remains 1, prior_doc intact)
    assert in_memory_db.query(DocumentModel).count() == 1
    retained_doc = in_memory_db.query(DocumentModel).first()
    assert retained_doc.id == "initial_doc_1"


def test_snapshot_concurrency_consistency():
    service = PipelineService()

    # Set initial snapshot (Dataset A)
    df_a = pd.DataFrame({"id": ["A1", "A2"], "clean_text": ["test a1", "test a2"], "sentiment": ["negative", "positive"], "aspects": [[], []], "source": ["Web", "Web"], "timestamp": [None, None]})
    snap_a = AnalysisSnapshot(
        df=df_a,
        clusters=[{"cluster_id": 0, "name": "Cluster A", "size": 2, "percentage": 100.0, "top_terms": ["a"], "center_2d": [0.0, 0.0]}],
        temporal_available=False,
        is_ready=True,
    )
    with service._lock:
        service._snapshot = snap_a

    # Reader thread captures snapshot A and holds reference
    reader_snapshot = service.get_snapshot()
    assert reader_snapshot.clusters[0]["name"] == "Cluster A"
    assert reader_snapshot.temporal_available is False

    # Concurrent writer updates service to Dataset B
    df_b = pd.DataFrame({"id": ["B1", "B2"], "clean_text": ["test b1", "test b2"], "sentiment": ["neutral", "neutral"], "aspects": [[], []], "source": ["App", "App"], "timestamp": [None, None]})
    snap_b = AnalysisSnapshot(
        df=df_b,
        clusters=[{"cluster_id": 0, "name": "Cluster B", "size": 2, "percentage": 100.0, "top_terms": ["b"], "center_2d": [1.0, 1.0]}],
        temporal_available=True,
        is_ready=True,
    )
    with service._lock:
        service._snapshot = snap_b

    # Reader holding reader_snapshot MUST continue seeing Dataset A consistently
    assert reader_snapshot.clusters[0]["name"] == "Cluster A"
    assert reader_snapshot.temporal_available is False
    assert len(reader_snapshot.df) == 2
    assert reader_snapshot.df["id"].iloc[0] == "A1"

    # New reader gets Dataset B
    new_reader = service.get_snapshot()
    assert new_reader.clusters[0]["name"] == "Cluster B"
    assert new_reader.temporal_available is True
    assert new_reader.df["id"].iloc[0] == "B1"


def test_temporal_unavailable_no_manufactured_surge():
    service = PipelineService()

    # Data without any timestamps
    no_time_df = pd.DataFrame([
        {"id": f"doc_{i}", "original_text": f"Negative complain issue {i}", "clean_text": f"Negative complain issue {i}", "normalized_text": f"negative complain issue {i}", "timestamp": None, "source": "Review", "language": "en", "word_count": 4, "metadata": {}}
        for i in range(25)
    ])

    res = service.run_full_pipeline(no_time_df, n_clusters=2, db=None)

    # Temporal must be marked unavailable
    assert res["temporal_available"] is False

    snap = service.get_snapshot()
    assert snap.temporal_available is False

    # Emerging issues must have recent_volume = 0 and surge_probability = None
    for em in snap.emerging:
        assert em["recent_volume"] == 0
        assert em["growth_rate_pct"] == 0.0
        assert em.get("surge_probability") is None
        assert em["data_sufficiency"] == "TEMPORAL_UNAVAILABLE"
        assert em["confidence_level"] == "NO_TEMPORAL_DATA"
