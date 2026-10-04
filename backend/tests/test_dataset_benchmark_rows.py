"""Permanent test comparing 10 real FRAUD rows and 10 real LEGIT rows from the original dataset.

Runs each sample through:
1. Direct notebook pipeline: scaler.transform -> model.predict_proba
2. Backend API feature pipeline: extract_features -> scaler.transform -> model.predict_proba

Asserts that both probabilities match exactly (difference < 1e-4).
"""

import pandas as pd
import pytest
from app.features import FEATURE_NAMES, extract_features
from app.model_service import model_service
from app.schemas import PredictRequest, TransactionType

# 10 Ground-Truth Real Fraud Rows from PaySim (clean_dataset.csv)
FRAUD_BENCHMARK_ROWS = [
    {
        "step": 1.0,
        "amount": 181.0,
        "oldbalanceOrg": 181.0,
        "newbalanceOrig": 0.0,
        "oldbalanceDest": 0.0,
        "newbalanceDest": 0.0,
        "errorBalanceOrig": 0.0,
        "errorBalanceDest": 181.0,
        "type_TRANSFER": 1.0,
    },
    {
        "step": 1.0,
        "amount": 2806.0,
        "oldbalanceOrg": 2806.0,
        "newbalanceOrig": 0.0,
        "oldbalanceDest": 0.0,
        "newbalanceDest": 0.0,
        "errorBalanceOrig": 0.0,
        "errorBalanceDest": 2806.0,
        "type_TRANSFER": 1.0,
    },
    {
        "step": 1.0,
        "amount": 20128.0,
        "oldbalanceOrg": 20128.0,
        "newbalanceOrig": 0.0,
        "oldbalanceDest": 0.0,
        "newbalanceDest": 0.0,
        "errorBalanceOrig": 0.0,
        "errorBalanceDest": 20128.0,
        "type_TRANSFER": 1.0,
    },
    {
        "step": 1.0,
        "amount": 35063.63,
        "oldbalanceOrg": 35063.63,
        "newbalanceOrig": 0.0,
        "oldbalanceDest": 0.0,
        "newbalanceDest": 0.0,
        "errorBalanceOrig": 0.0,
        "errorBalanceDest": 35063.63,
        "type_TRANSFER": 1.0,
    },
    {
        "step": 1.0,
        "amount": 25071.46,
        "oldbalanceOrg": 25071.46,
        "newbalanceOrig": 0.0,
        "oldbalanceDest": 0.0,
        "newbalanceDest": 0.0,
        "errorBalanceOrig": 0.0,
        "errorBalanceDest": 25071.46,
        "type_TRANSFER": 1.0,
    },
    {
        "step": 1.0,
        "amount": 25071.46,
        "oldbalanceOrg": 25071.46,
        "newbalanceOrig": 0.0,
        "oldbalanceDest": 9083.76,
        "newbalanceDest": 34155.22,
        "errorBalanceOrig": 0.0,
        "errorBalanceDest": 0.0,
        "type_TRANSFER": 0.0,
    },
    {
        "step": 1.0,
        "amount": 235238.66,
        "oldbalanceOrg": 235238.66,
        "newbalanceOrig": 0.0,
        "oldbalanceDest": 0.0,
        "newbalanceDest": 0.0,
        "errorBalanceOrig": 0.0,
        "errorBalanceDest": 235238.66,
        "type_TRANSFER": 1.0,
    },
    {
        "step": 2.0,
        "amount": 963532.14,
        "oldbalanceOrg": 963532.14,
        "newbalanceOrig": 0.0,
        "oldbalanceDest": 0.0,
        "newbalanceDest": 0.0,
        "errorBalanceOrig": 0.0,
        "errorBalanceDest": 963532.14,
        "type_TRANSFER": 1.0,
    },
    {
        "step": 2.0,
        "amount": 963532.14,
        "oldbalanceOrg": 963532.14,
        "newbalanceOrig": 0.0,
        "oldbalanceDest": 132382.57,
        "newbalanceDest": 1095914.71,
        "errorBalanceOrig": 0.0,
        "errorBalanceDest": 0.0,
        "type_TRANSFER": 0.0,
    },
    {
        "step": 2.0,
        "amount": 14949.84,
        "oldbalanceOrg": 14949.84,
        "newbalanceOrig": 0.0,
        "oldbalanceDest": 0.0,
        "newbalanceDest": 0.0,
        "errorBalanceOrig": 0.0,
        "errorBalanceDest": 14949.84,
        "type_TRANSFER": 1.0,
    },
]

# 10 Ground-Truth Real Legit Rows from PaySim (clean_dataset.csv)
LEGIT_BENCHMARK_ROWS = [
    {
        "step": 1.0,
        "amount": 6039.61,
        "oldbalanceOrg": 0.0,
        "newbalanceOrig": 0.0,
        "oldbalanceDest": 31164.0,
        "newbalanceDest": 37203.61,
        "errorBalanceOrig": 6039.61,
        "errorBalanceDest": 0.0,
        "type_TRANSFER": 0.0,
    },
    {
        "step": 1.0,
        "amount": 9104.26,
        "oldbalanceOrg": 0.0,
        "newbalanceOrig": 0.0,
        "oldbalanceDest": 10623.0,
        "newbalanceDest": 19727.26,
        "errorBalanceOrig": 9104.26,
        "errorBalanceDest": 0.0,
        "type_TRANSFER": 0.0,
    },
    {
        "step": 1.0,
        "amount": 11136.72,
        "oldbalanceOrg": 0.0,
        "newbalanceOrig": 0.0,
        "oldbalanceDest": 16180.11,
        "newbalanceDest": 27316.83,
        "errorBalanceOrig": 11136.72,
        "errorBalanceDest": 0.0,
        "type_TRANSFER": 1.0,
    },
    {
        "step": 1.0,
        "amount": 19772.53,
        "oldbalanceOrg": 0.0,
        "newbalanceOrig": 0.0,
        "oldbalanceDest": 44486.99,
        "newbalanceDest": 64259.52,
        "errorBalanceOrig": 19772.53,
        "errorBalanceDest": 0.0,
        "type_TRANSFER": 1.0,
    },
    {
        "step": 1.0,
        "amount": 271161.74,
        "oldbalanceOrg": 0.0,
        "newbalanceOrig": 0.0,
        "oldbalanceDest": 0.0,
        "newbalanceDest": 0.0,
        "errorBalanceOrig": 271161.74,
        "errorBalanceDest": 271161.74,
        "type_TRANSFER": 1.0,
    },
    {
        "step": 1.0,
        "amount": 40149.38,
        "oldbalanceOrg": 287443.9,
        "newbalanceOrig": 247294.52,
        "oldbalanceDest": 26426.12,
        "newbalanceDest": 66575.5,
        "errorBalanceOrig": 0.0,
        "errorBalanceDest": 0.0,
        "type_TRANSFER": 1.0,
    },
    {
        "step": 1.0,
        "amount": 94527.39,
        "oldbalanceOrg": 0.0,
        "newbalanceOrig": 0.0,
        "oldbalanceDest": 95007.35,
        "newbalanceDest": 189534.74,
        "errorBalanceOrig": 94527.39,
        "errorBalanceDest": 0.0,
        "type_TRANSFER": 0.0,
    },
    {
        "step": 1.0,
        "amount": 20986.8,
        "oldbalanceOrg": 0.0,
        "newbalanceOrig": 0.0,
        "oldbalanceDest": 25475.43,
        "newbalanceDest": 46462.23,
        "errorBalanceOrig": 20986.8,
        "errorBalanceDest": 0.0,
        "type_TRANSFER": 0.0,
    },
    {
        "step": 1.0,
        "amount": 86401.93,
        "oldbalanceOrg": 0.0,
        "newbalanceOrig": 0.0,
        "oldbalanceDest": 879468.12,
        "newbalanceDest": 965870.05,
        "errorBalanceOrig": 86401.93,
        "errorBalanceDest": 0.0,
        "type_TRANSFER": 1.0,
    },
    {
        "step": 1.0,
        "amount": 90667.11,
        "oldbalanceOrg": 0.0,
        "newbalanceOrig": 0.0,
        "oldbalanceDest": 411812.91,
        "newbalanceDest": 502480.02,
        "errorBalanceOrig": 90667.11,
        "errorBalanceDest": 0.0,
        "type_TRANSFER": 1.0,
    },
]


@pytest.fixture(scope="module", autouse=True)
def ensure_artifacts_loaded():
    """Ensure model service artifacts are loaded."""
    if not model_service.is_loaded:
        model_service.load_artifacts()


def test_10_fraud_rows_pipeline_match():
    """Test 10 real fraud dataset rows through Direct and API pipelines side-by-side."""
    print("\n--- 10 Real Fraud Rows Comparison ---")
    for i, raw_dict in enumerate(FRAUD_BENCHMARK_ROWS):
        # 1. Direct Pipeline
        direct_df = pd.DataFrame([raw_dict])[FEATURE_NAMES]
        direct_scaled = model_service.scaler.transform(direct_df)
        direct_prob = float(model_service.model.predict_proba(direct_scaled)[0, 1])

        # 2. API Pipeline
        tx_type = (
            TransactionType.TRANSFER
            if raw_dict["type_TRANSFER"] == 1.0
            else TransactionType.CASH_OUT
        )
        req = PredictRequest(
            type=tx_type,
            amount=raw_dict["amount"],
            sender_balance=raw_dict["oldbalanceOrg"],
            recipient_balance=raw_dict["oldbalanceDest"],
            time=f"{int(raw_dict['step']):02d}:00",
        )
        api_df = extract_features(req)
        api_prob, _ = model_service.predict_risk(api_df)

        diff = abs(direct_prob - api_prob)
        print(
            f"Fraud Row {i+1:>2}: Direct Prob={direct_prob:.4f} | "
            f"API Prob={api_prob:.4f} | Diff={diff:.6f}"
        )
        assert diff < 1e-4, f"Mismatch on fraud row {i+1}: direct={direct_prob}, api={api_prob}"


def test_10_legit_rows_pipeline_match():
    """Test 10 real legit dataset rows through Direct and API pipelines side-by-side."""
    print("\n--- 10 Real Legit Rows Comparison ---")
    for i, raw_dict in enumerate(LEGIT_BENCHMARK_ROWS):
        # 1. Direct Pipeline
        direct_df = pd.DataFrame([raw_dict])[FEATURE_NAMES]
        direct_scaled = model_service.scaler.transform(direct_df)
        direct_prob = float(model_service.model.predict_proba(direct_scaled)[0, 1])

        # 2. API Pipeline
        tx_type = (
            TransactionType.TRANSFER
            if raw_dict["type_TRANSFER"] == 1.0
            else TransactionType.CASH_OUT
        )
        req = PredictRequest(
            type=tx_type,
            amount=raw_dict["amount"],
            sender_balance=raw_dict["oldbalanceOrg"],
            recipient_balance=raw_dict["oldbalanceDest"],
            time=f"{int(raw_dict['step']):02d}:00",
        )
        api_df = extract_features(req)
        api_prob, _ = model_service.predict_risk(api_df)

        diff = abs(direct_prob - api_prob)
        print(
            f"Legit Row {i+1:>2}: Direct Prob={direct_prob:.4f} | "
            f"API Prob={api_prob:.4f} | Diff={diff:.6f}"
        )
        assert diff < 1e-4, f"Mismatch on legit row {i+1}: direct={direct_prob}, api={api_prob}"
