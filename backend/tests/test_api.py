import pytest
from fastapi.testclient import TestClient
from backend.app.main import app

client = TestClient(app)


def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["project"] == "VoiceMiner"


def test_dashboard_endpoint():
    response = client.get("/api/dashboard")
    assert response.status_code == 200
    data = response.json()
    assert "total_feedback" in data
    assert "sentiment_distribution" in data
    assert "top_aspects" in data


def test_issues_endpoints():
    response = client.get("/api/issues")
    assert response.status_code == 200
    issues = response.json()
    assert isinstance(issues, list)
    if issues:
        first_id = issues[0]["issue_id"]
        detail_res = client.get(f"/api/issues/{first_id}")
        assert detail_res.status_code == 200
        root_causes_res = client.get(f"/api/issues/{first_id}/root-causes")
        assert root_causes_res.status_code == 200
        evidence_res = client.get(f"/api/issues/{first_id}/evidence")
        assert evidence_res.status_code == 200


def test_emerging_issues_endpoint():
    response = client.get("/api/emerging-issues")
    assert response.status_code == 200
    emerging = response.json()
    assert isinstance(emerging, list)


def test_graph_endpoint():
    response = client.get("/api/graph")
    assert response.status_code == 200
    graph = response.json()
    assert "nodes" in graph
    assert "links" in graph


def test_text_analyzer_endpoint():
    payload = {"text": "Flight was delayed 4 hours and customer service was unresponsive!"}
    response = client.post("/api/text/analyze", json=payload)
    assert response.status_code == 200
    res = response.json()
    assert "sentiment" in res
    assert "emotion" in res
    assert "aspects" in res


def test_reports_and_evaluation_endpoints():
    rep_res = client.post("/api/reports/generate", json={"include_raw_evidence": True})
    assert rep_res.status_code == 200
    rep_data = rep_res.json()
    assert "executive_summary" in rep_data
    assert "candidate_root_causes" in rep_data
    assert "limitations" in rep_data

    metrics_res = client.get("/api/evaluation/metrics")
    assert metrics_res.status_code == 200
    metrics_data = metrics_res.json()
    assert "clustering_benchmark" in metrics_data
