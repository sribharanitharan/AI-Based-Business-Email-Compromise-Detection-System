"""
test_api.py — Integration tests for FastAPI endpoints
"""

import pytest
from fastapi.testclient import TestClient
from app import app


@pytest.fixture
def client():
    return TestClient(app)


def test_health_endpoint(client):
    response = client.get("/api/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] in ["ok", "degraded"]
    assert "models_loaded" in data


def test_analyze_endpoint(client):
    payload = {
        "subject": "Urgent: Direct wire payment required",
        "sender": "ceo@external-fraud.xyz",
        "reply_to": "attacker@gmail.com",
        "body": "Please wire $75,000 to our supplier account immediately.",
        "headers": {"spf": "fail"}
    }
    response = client.post("/api/analyze", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "risk_score" in data
    assert "risk_level" in data
    assert "classification" in data
    assert "flags" in data


def test_batch_analyze(client):
    emails = [
        {
            "subject": "Lunch meeting tomorrow?",
            "sender": "colleague@company.com",
            "body": "Hey, do you want to grab lunch around noon?"
        },
        {
            "subject": "Urgent payment request",
            "sender": "vendor@fake.com",
            "body": "Please process payment for the overdue invoice."
        }
    ]
    response = client.post("/api/analyze/batch", json=emails)
    assert response.status_code == 200
    data = response.json()
    assert data["total"] == 2
    assert len(data["results"]) == 2


def test_stats_and_history(client):
    hist_resp = client.get("/api/history")
    assert hist_resp.status_code == 200
    assert isinstance(hist_resp.json(), list)

    stats_resp = client.get("/api/stats")
    assert stats_resp.status_code == 200
    stats = stats_resp.json()
    assert "total_analyzed" in stats
    assert "bec_detected" in stats


def test_threat_intel_endpoint(client):
    response = client.get("/api/threat-intel")
    assert response.status_code == 200
    data = response.json()
    assert "feeds" in data
