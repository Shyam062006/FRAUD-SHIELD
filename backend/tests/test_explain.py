"""Unit tests for explanation generation engine."""

import numpy as np
import pytest
from app.explain import generate_reasons
from app.schemas import SignalType, TransactionType


def test_reasons_contain_all_required_signals():
    """Verify explanations cover all 4 defined telemetry signals."""
    importances = np.full(9, 1.0 / 9.0)
    reasons = generate_reasons(
        tx_type=TransactionType.TRANSFER,
        amount=250.0,
        sender_balance=5000.0,
        recipient_balance=3200.0,
        time_str="14:30",
        fraud_risk=0.01,
        feature_importances=importances,
    )

    signals_present = {r.signal for r in reasons}
    assert SignalType.BALANCE_BEHAVIOR in signals_present
    assert SignalType.TIME_PATTERN in signals_present
    assert SignalType.TRANSACTION_TYPE in signals_present
    assert SignalType.AMOUNT_PATTERN in signals_present


def test_reasons_severity_ordering():
    """Verify reasons are sorted with highest severity levels first."""
    importances = np.full(9, 1.0 / 9.0)
    # Severe transaction: late night, large depletion, cashout
    reasons = generate_reasons(
        tx_type=TransactionType.CASH_OUT,
        amount=9800.0,
        sender_balance=10000.0,
        recipient_balance=0.0,
        time_str="02:15",
        fraud_risk=0.85,
        feature_importances=importances,
    )

    levels = [r.level for r in reasons]
    # Check descending order:
    assert levels == sorted(levels, reverse=True)
    assert 2 in levels


def test_no_raw_feature_names_in_explanations():
    """Verify that internal column names are never exposed in user-facing text."""
    raw_feature_names = [
        "step",
        "oldbalanceOrg",
        "newbalanceOrig",
        "oldbalanceDest",
        "newbalanceDest",
        "errorBalanceOrig",
        "errorBalanceDest",
        "type_TRANSFER",
    ]
    importances = np.full(9, 1.0 / 9.0)
    reasons = generate_reasons(
        tx_type=TransactionType.CASH_OUT,
        amount=9800.0,
        sender_balance=10000.0,
        recipient_balance=0.0,
        time_str="02:15",
        fraud_risk=0.85,
        feature_importances=importances,
    )

    for r in reasons:
        for raw in raw_feature_names:
            assert raw not in r.text, f"Raw feature '{raw}' leaked in user text: '{r.text}'"


def test_high_depletion_triggers_level_2():
    """Verify that moving >=80% of sender's balance triggers level 2 warning."""
    importances = np.full(9, 1.0 / 9.0)
    reasons = generate_reasons(
        tx_type=TransactionType.TRANSFER,
        amount=9500.0,
        sender_balance=10000.0,
        recipient_balance=5000.0,
        time_str="12:00",
        fraud_risk=0.5,
        feature_importances=importances,
    )
    bal_reasons = [r for r in reasons if r.signal == SignalType.BALANCE_BEHAVIOR]
    assert any(r.level == 2 for r in bal_reasons)
    assert any("Most of the sender's balance" in r.text for r in bal_reasons)


def test_night_time_triggers_level_2():
    """Verify that late night/early morning hours trigger level 2 warning."""
    importances = np.full(9, 1.0 / 9.0)
    reasons = generate_reasons(
        tx_type=TransactionType.TRANSFER,
        amount=100.0,
        sender_balance=5000.0,
        recipient_balance=5000.0,
        time_str="02:15",
        fraud_risk=0.1,
        feature_importances=importances,
    )
    time_reasons = [r for r in reasons if r.signal == SignalType.TIME_PATTERN]
    assert any(r.level == 2 for r in time_reasons)
    assert any("late at night" in r.text for r in time_reasons)
