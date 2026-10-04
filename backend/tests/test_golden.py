"""Golden tests: verify API probability exactly matches direct model.predict_proba execution."""

import joblib
import pytest
from fastapi.testclient import TestClient
from app.config import settings
from app.features import extract_features
from app.main import app
from app.schemas import PredictRequest, TransactionType


@pytest.fixture(scope="module")
def artifacts():
    """Load model and scaler artifacts directly from disk."""
    model = joblib.load(settings.resolved_model_path)
    scaler = joblib.load(settings.resolved_scaler_path)
    return model, scaler


@pytest.fixture(scope="module")
def client():
    """FastAPI TestClient with lifespan context manager."""
    with TestClient(app) as test_client:
        yield test_client


GOLDEN_INPUTS = [
    {
        "type": "TRANSFER",
        "amount": 250.0,
        "sender_balance": 5000.0,
        "recipient_balance": 3200.0,
        "time": "14:30",
    },
    {
        "type": "CASH_OUT",
        "amount": 9800.0,
        "sender_balance": 10000.0,
        "recipient_balance": 0.0,
        "time": "02:15",
    },
    {
        "type": "TRANSFER",
        "amount": 150000.0,
        "sender_balance": 200000.0,
        "recipient_balance": 50000.0,
        "time": "09:15",
    },
    {
        "type": "CASH_OUT",
        "amount": 50000.0,
        "sender_balance": 50000.0,
        "recipient_balance": 1000.0,
        "time": "23:30",
    },
    {
        "type": "TRANSFER",
        "amount": 10.0,
        "sender_balance": 1000000.0,
        "recipient_balance": 500000.0,
        "time": "12:00",
    },
]


@pytest.mark.parametrize("payload", GOLDEN_INPUTS)
def test_api_probability_matches_direct_model_prediction(client, artifacts, payload):
    """Assert that the API risk probability matches direct model.predict_proba on scaled features."""
    direct_model, direct_scaler = artifacts

    # 1. Direct computation
    req = PredictRequest(**payload)
    features_df = extract_features(req)
    scaled_features = direct_scaler.transform(features_df)
    direct_probs = direct_model.predict_proba(scaled_features)
    direct_risk = float(direct_probs[0, 1])

    # 2. API endpoint execution
    response = client.post("/api/predict", json=payload)
    assert response.status_code == 200
    api_data = response.json()
    api_risk = api_data["risk"]

    # 3. Assert golden equivalence (API rounds to 4 decimal places)
    assert api_risk == pytest.approx(round(direct_risk, 4), abs=1e-4)
