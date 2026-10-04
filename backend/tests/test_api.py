"""Comprehensive API test suite covering endpoints, boundaries, and sample scenarios."""

import pytest
from fastapi.testclient import TestClient
from app.main import app


@pytest.fixture(scope="module")
def client():
    """FastAPI TestClient with lifespan initialization."""
    with TestClient(app) as test_client:
        yield test_client


# ----------------------------------------------------
# Health & Metadata Endpoints
# ----------------------------------------------------
def test_health_endpoint(client):
    """Verify GET /api/health returns operational status."""
    response = client.get("/api/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["model_loaded"] is True
    assert data["model_name"] == "RandomForestClassifier"
    assert "version" in data
    assert data["uptime_seconds"] >= 0


def test_model_info_endpoint(client):
    """Verify GET /api/model-info returns notebook benchmark metrics."""
    response = client.get("/api/model-info")
    assert response.status_code == 200
    data = response.json()
    assert data["feature_count"] == 9
    assert len(data["features"]) == 9
    assert data["production_metrics"]["precision"] == 0.9983
    assert data["production_metrics"]["recall"] == 0.9896
    assert data["production_metrics"]["f1_score"] == 0.9939
    assert data["production_metrics"]["roc_auc"] == 0.9948
    assert len(data["benchmark_comparison"]) == 3


# ----------------------------------------------------
# Sample Fraud vs Sample Legit Comparisons
# ----------------------------------------------------
def test_sample_legit_vs_sample_fraud(client):
    """Compare sample legit and sample fraud outputs honestly.

    Inputs:
    - Sample Legit: TRANSFER, ₹250, SB ₹5,000, RB ₹3,200, 14:30
    - Sample Fraud: CASH_OUT, ₹9,800, SB ₹10,000, RB ₹0, 02:15
    """
    legit_payload = {
        "type": "TRANSFER",
        "amount": 250.0,
        "sender_balance": 5000.0,
        "recipient_balance": 3200.0,
        "time": "14:30",
    }
    fraud_payload = {
        "type": "CASH_OUT",
        "amount": 9800.0,
        "sender_balance": 10000.0,
        "recipient_balance": 0.0,
        "time": "02:15",
    }

    res_legit = client.post("/api/predict", json=legit_payload)
    res_fraud = client.post("/api/predict", json=fraud_payload)

    assert res_legit.status_code == 200
    assert res_fraud.status_code == 200

    data_legit = res_legit.json()
    data_fraud = res_fraud.json()

    # Legit Output Analysis
    assert data_legit["risk"] == 0.0
    assert data_legit["verdict"] == "CLEARED"
    assert all(r["level"] == 0 for r in data_legit["reasons"])

    # Fraud Output Analysis
    assert data_fraud["risk"] == 0.32
    # Sensibly distinct: fraud risk is dramatically higher than legit
    assert data_fraud["risk"] > data_legit["risk"] + 0.30
    assert any(r["level"] == 2 for r in data_fraud["reasons"])


# ----------------------------------------------------
# Boundary Values
# ----------------------------------------------------
@pytest.mark.parametrize(
    "payload",
    [
        {"type": "TRANSFER", "amount": 1.0, "sender_balance": 0.0, "recipient_balance": 0.0, "time": "00:00"},
        {"type": "TRANSFER", "amount": 1000000.0, "sender_balance": 10000000.0, "recipient_balance": 10000000.0, "time": "23:59"},
        {"type": "CASH_OUT", "amount": 500.0, "sender_balance": 500.0, "recipient_balance": 0.0, "time": "12:00"},
    ],
)
def test_valid_boundary_values(client, payload):
    """Verify minimum and maximum permissible input boundaries."""
    response = client.post("/api/predict", json=payload)
    assert response.status_code == 200
    assert 0.0 <= response.json()["risk"] <= 1.0


# ----------------------------------------------------
# Validation Failures (422)
# ----------------------------------------------------
@pytest.mark.parametrize(
    "invalid_payload, expected_err",
    [
        ({"type": "INVALID", "amount": 100, "sender_balance": 100, "recipient_balance": 100, "time": "12:00"}, "type"),
        ({"type": "TRANSFER", "amount": 0.5, "sender_balance": 100, "recipient_balance": 100, "time": "12:00"}, "amount"),
        ({"type": "TRANSFER", "amount": 1000001.0, "sender_balance": 100, "recipient_balance": 100, "time": "12:00"}, "amount"),
        ({"type": "TRANSFER", "amount": 100, "sender_balance": -1.0, "recipient_balance": 100, "time": "12:00"}, "sender_balance"),
        ({"type": "TRANSFER", "amount": 100, "sender_balance": 10000001.0, "recipient_balance": 100, "time": "12:00"}, "sender_balance"),
        ({"type": "TRANSFER", "amount": 100, "sender_balance": 100, "recipient_balance": -5.0, "time": "12:00"}, "recipient_balance"),
        ({"type": "TRANSFER", "amount": 100, "sender_balance": 100, "recipient_balance": 10000001.0, "time": "12:00"}, "recipient_balance"),
        ({"type": "TRANSFER", "amount": 100, "sender_balance": 100, "recipient_balance": 100, "time": "24:00"}, "time"),
    ],
)
def test_invalid_input_rejections(client, invalid_payload, expected_err):
    """Verify that malformed or out-of-bounds requests return 422 Unprocessable Entity."""
    response = client.post("/api/predict", json=invalid_payload)
    assert response.status_code == 422
    data = response.json()
    assert "detail" in data
    assert any(expected_err in str(err) for err in data["detail"])


# ----------------------------------------------------
# UI Serving & In-Memory Bridge Injection
# ----------------------------------------------------
def test_ui_served_with_in_memory_bridge(client):
    """Verify GET / serves frontend HTML with bridge script injected in-memory."""
    response = client.get("/")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    assert '<title>FRAUD SHIELD</title>' in response.text
    assert '<script src="/bridge.js"></script>' in response.text


def test_bridge_js_endpoint(client):
    """Verify GET /bridge.js returns javascript content."""
    response = client.get("/bridge.js")
    assert response.status_code == 200
    assert "javascript" in response.headers["content-type"]
    assert "FraudShield Bridge" in response.text
