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


def test_api_emerging_issues_schema_with_populated_analysis():
    """
    Validates that /api/emerging-issues returns valid EmergingIssueItem objects
    with both issue_id and cluster_id matching without validation errors.
    """
    from fastapi.testclient import TestClient
    from backend.app.main import app
    from backend.app.schemas.schemas import EmergingIssueItem

    service = PipelineService.get_instance()
    df = pd.DataFrame([
        {"id": f"t_{i}", "original_text": f"Customer delay and flight issue {i}", "clean_text": f"Customer delay and flight issue {i}", "normalized_text": f"customer delay flight issue {i}", "timestamp": None, "source": "Twitter", "language": "en", "word_count": 6, "metadata": {}}
        for i in range(20)
    ])
    service.run_full_pipeline(df, n_clusters=2, db=None)

    client = TestClient(app)
    response = client.get("/api/emerging-issues")
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)
    assert len(data) == 2

    for item in data:
        # Validate through Pydantic model directly
        validated = EmergingIssueItem(**item)
        assert validated.issue_id == item["issue_id"]
        assert validated.cluster_id == item["issue_id"]
        assert isinstance(validated.emerging_score, float)
        assert validated.priority in {"CRITICAL", "HIGH", "MEDIUM", "LOW"}


def test_genuine_multithreaded_realtime_matching_concurrency():
    """
    Genuine multi-threaded regression test verifying that real-time text matching
    operates reliably against snapshot-isolated models while a concurrent thread
    re-runs pipeline analysis on a separate dataset.
    """
    from concurrent.futures import ThreadPoolExecutor

    service = PipelineService()

    # Pre-populate with Dataset 1 (2 clusters)
    df_1 = pd.DataFrame([
        {"id": f"d1_{i}", "original_text": f"Payment gateway crash bug {i}", "clean_text": f"Payment gateway crash bug {i}", "normalized_text": f"payment gateway crash bug {i}", "timestamp": None, "source": "App", "language": "en", "word_count": 5, "metadata": {}}
        for i in range(20)
    ])
    service.run_full_pipeline(df_1, n_clusters=2, db=None)
    assert service.get_snapshot().is_ready is True

    errors = []
    realtime_results = []
    stop_event = threading.Event()

    def realtime_reader():
        while not stop_event.is_set():
            try:
                res = service.analyze_realtime_text("The payment failed and app crashed")
                assert res is not None
                assert "matched_issue" in res
                realtime_results.append(res)
                time.sleep(0.01)
            except Exception as e:
                errors.append(e)

    def pipeline_writer():
        # Re-run pipeline with Dataset 2 (3 clusters)
        df_2 = pd.DataFrame([
            {"id": f"d2_{i}", "original_text": f"Flight delay lost baggage carousel {i}", "clean_text": f"Flight delay lost baggage carousel {i}", "normalized_text": f"flight delay lost baggage carousel {i}", "timestamp": None, "source": "Web", "language": "en", "word_count": 6, "metadata": {}}
            for i in range(25)
        ])
        time.sleep(0.03)  # let reader run a few times on snapshot 1
        service.run_full_pipeline(df_2, n_clusters=3, db=None)

    with ThreadPoolExecutor(max_workers=2) as executor:
        f_reader = executor.submit(realtime_reader)
        f_writer = executor.submit(pipeline_writer)

        f_writer.result()  # wait for pipeline to complete
        time.sleep(0.05)   # let reader execute a few more iterations on the new snapshot
        stop_event.set()
        f_reader.result()

    # Zero exceptions must have occurred across all threads
    assert len(errors) == 0, f"Concurrency errors encountered: {errors}"
    assert len(realtime_results) >= 5, "Expected multiple successful real-time analyses during concurrency"

    # Verify final snapshot is updated to Dataset 2
    final_snap = service.get_snapshot()
    assert len(final_snap.clusters) == 3
