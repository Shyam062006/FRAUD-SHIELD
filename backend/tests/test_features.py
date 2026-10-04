"""Unit tests for feature engineering and extraction."""

import pytest
import pandas as pd
from app.features import FEATURE_NAMES, extract_features, parse_time_to_step
from app.schemas import PredictRequest, TransactionType


def test_feature_names_order_and_count():
    """Verify that feature names match the exact 9 features in training order."""
    expected = [
        "step",
        "amount",
        "oldbalanceOrg",
        "newbalanceOrig",
        "oldbalanceDest",
        "newbalanceDest",
        "errorBalanceOrig",
        "errorBalanceDest",
        "type_TRANSFER",
    ]
    assert FEATURE_NAMES == expected
    assert len(FEATURE_NAMES) == 9


def test_parse_time_to_step():
    """Verify parsing time string to fractional step hours."""
    assert parse_time_to_step("00:00") == pytest.approx(0.0)
    assert parse_time_to_step("14:30") == pytest.approx(14.5)
    assert parse_time_to_step("02:15") == pytest.approx(2.25)
    assert parse_time_to_step("23:45") == pytest.approx(23.75)


def test_extract_features_transfer():
    """Verify feature mapping for a standard TRANSFER transaction."""
    req = PredictRequest(
        type=TransactionType.TRANSFER,
        amount=500.0,
        sender_balance=2000.0,
        recipient_balance=1000.0,
        time="10:30",
    )
    df = extract_features(req)

    assert isinstance(df, pd.DataFrame)
    assert list(df.columns) == FEATURE_NAMES
    assert df.loc[0, "step"] == pytest.approx(10.5)
    assert df.loc[0, "amount"] == pytest.approx(500.0)
    assert df.loc[0, "oldbalanceOrg"] == pytest.approx(2000.0)
    assert df.loc[0, "newbalanceOrig"] == pytest.approx(1500.0)
    assert df.loc[0, "oldbalanceDest"] == pytest.approx(1000.0)
    assert df.loc[0, "newbalanceDest"] == pytest.approx(1500.0)
    assert df.loc[0, "errorBalanceOrig"] == pytest.approx(0.0)
    assert df.loc[0, "errorBalanceDest"] == pytest.approx(0.0)
    assert df.loc[0, "type_TRANSFER"] == pytest.approx(1.0)


def test_extract_features_cash_out_zero_dest():
    """Verify feature mapping for CASH_OUT where recipient is an untracked cashout sink."""
    req = PredictRequest(
        type=TransactionType.CASH_OUT,
        amount=9800.0,
        sender_balance=10000.0,
        recipient_balance=0.0,
        time="02:15",
    )
    df = extract_features(req)

    assert df.loc[0, "step"] == pytest.approx(2.25)
    assert df.loc[0, "amount"] == pytest.approx(9800.0)
    assert df.loc[0, "oldbalanceOrg"] == pytest.approx(10000.0)
    assert df.loc[0, "newbalanceOrig"] == pytest.approx(200.0)
    assert df.loc[0, "oldbalanceDest"] == pytest.approx(0.0)
    assert df.loc[0, "newbalanceDest"] == pytest.approx(0.0)
    assert df.loc[0, "errorBalanceOrig"] == pytest.approx(0.0)
    assert df.loc[0, "errorBalanceDest"] == pytest.approx(9800.0)
    assert df.loc[0, "type_TRANSFER"] == pytest.approx(0.0)


def test_extract_features_overdraft_error_balance():
    """Verify errorBalanceOrig when amount exceeds starting balance."""
    req = PredictRequest(
        type=TransactionType.TRANSFER,
        amount=5000.0,
        sender_balance=2000.0,
        recipient_balance=500.0,
        time="12:00",
    )
    df = extract_features(req)
    # newbalanceOrig is capped at 0.0
    assert df.loc[0, "newbalanceOrig"] == pytest.approx(0.0)
    # errorBalanceOrig = 0.0 + 5000 - 2000 = 3000
    assert df.loc[0, "errorBalanceOrig"] == pytest.approx(3000.0)
