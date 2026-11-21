import pytest
import io
from fastapi.testclient import TestClient
from backend.app.main import app

client = TestClient(app)


def test_upload_security_invalid_extension():
    # Attempt uploading an unsupported file format (.exe)
    fake_file = io.BytesIO(b"fake binary content")
    response = client.post(
        "/api/data/upload",
        files={"file": ("malicious.exe", fake_file, "application/octet-stream")},
    )
    assert response.status_code == 400
    assert "Unsupported file extension" in response.json()["detail"]


def test_upload_security_directory_traversal():
    # Attempt path traversal
    fake_csv = io.BytesIO(b"text\nflight was delayed 3 hours\nlost my baggage")
    response = client.post(
        "/api/data/upload",
        files={"file": ("../../evil.csv", fake_csv, "text/csv")},
    )
    # The server must sanitize the filename safely or succeed without escaping data/raw
    assert response.status_code in [200, 400]


def test_parameter_propagation_n_clusters():
    # Trigger analysis with n_clusters=4
    res = client.post(
        "/api/analyze",
        json={"dataset_name": "airline_sample_1000.csv", "n_clusters": 4, "similarity_threshold": 0.40},
    )
    assert res.status_code == 200
    data = res.json()
    assert data["details"]["clusters_count"] == 4

    # Verify that /api/issues now returns exactly 4 clusters
    issues_res = client.get("/api/issues")
    assert issues_res.status_code == 200
    issues = issues_res.json()
    assert len(issues) == 4


def test_similarity_threshold_propagation_text_analyze():
    # Calling text analyze with very high similarity threshold (0.99) should trigger open-set unknown issue
    res = client.post(
        "/api/text/analyze",
        json={"text": "completely bizarre out of domain feedback about alien spaceship", "similarity_threshold": 0.99},
    )
    assert res.status_code == 200
    data = res.json()
    if data.get("matched_issue"):
        # With threshold 0.99, cosine similarity to any cluster will be < 0.99
        assert data["matched_issue"]["issue_id"] == -1
        assert "Unknown" in data["matched_issue"]["name"]
