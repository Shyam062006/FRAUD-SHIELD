"""Unit tests for explanation generation engine."""

import numpy as np
import pytest
from app.explain import generate_reasons
from app.schemas import SignalType, TransactionType


def test_reasons_contain_all_required_signals():
    """Verify explanations cover all 4 defined telemetry signals."""
    dummy_contribs = np.zeros(9)
    effective_risk, verdict, reasons = generate_reasons(
        tx_type=TransactionType.TRANSFER,
        amount=250.0,
        sender_balance=5000.0,
        recipient_balance=3200.0,
        time_str="14:30",
        raw_model_risk=0.01,
        threshold=0.50,
        contributions=dummy_contribs,
        is_impossible=False,
    )

    signals_present = {r.signal for r in reasons}
    assert SignalType.BALANCE_BEHAVIOR in signals_present
    assert SignalType.TIME_PATTERN in signals_present
    assert SignalType.TRANSACTION_TYPE in signals_present
    assert SignalType.AMOUNT_PATTERN in signals_present


def test_reasons_severity_ordering():
    """Verify reasons are sorted with highest severity levels first."""
    # Strong contributions on balance and time
    contribs = np.array([0.15, 0.0, 0.05, 0.05, 0.0, 0.0, 0.0, 0.0, 0.0])
    effective_risk, verdict, reasons = generate_reasons(
        tx_type=TransactionType.CASH_OUT,
        amount=9800.0,
        sender_balance=10000.0,
        recipient_balance=0.0,
        time_str="02:15",
        raw_model_risk=0.85,
        threshold=0.50,
        contributions=contribs,
        is_impossible=False,
    )

    levels = [r.level for r in reasons]
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
    contribs = np.array([0.1, 0.1, 0.05, 0.05, 0.0, 0.0, 0.0, 0.0, 0.0])
    effective_risk, verdict, reasons = generate_reasons(
        tx_type=TransactionType.CASH_OUT,
        amount=9800.0,
        sender_balance=10000.0,
        recipient_balance=0.0,
        time_str="02:15",
        raw_model_risk=0.85,
        threshold=0.50,
        contributions=contribs,
        is_impossible=False,
    )

    for r in reasons:
        for raw in raw_feature_names:
            assert raw not in r.text, f"Raw feature '{raw}' leaked in user text: '{r.text}'"


def test_high_depletion_triggers_level_2_when_flagged():
    """Verify that moving >=80% of sender's balance with positive contribution triggers level 2 warning when flagged."""
    contribs = np.zeros(9)
    contribs[2] = 0.09  # balance contribution
    effective_risk, verdict, reasons = generate_reasons(
        tx_type=TransactionType.TRANSFER,
        amount=9500.0,
        sender_balance=10000.0,
        recipient_balance=5000.0,
        time_str="12:00",
        raw_model_risk=0.85,
        threshold=0.50,
        contributions=contribs,
        is_impossible=False,
    )
    bal_reasons = [r for r in reasons if r.signal == SignalType.BALANCE_BEHAVIOR]
    assert any(r.level == 2 for r in bal_reasons)
    assert any("Most of the sender's balance" in r.text for r in bal_reasons)


def test_cleared_verdict_strictly_prevents_level_2():
    """Verify Rule 2: If verdict is CLEARED, no reason may be level 2."""
    contribs = np.full(9, 0.1)  # High contributions across the board
    effective_risk, verdict, reasons = generate_reasons(
        tx_type=TransactionType.TRANSFER,
        amount=100.0,
        sender_balance=5000.0,
        recipient_balance=5000.0,
        time_str="02:15",
        raw_model_risk=0.20,  # Below 0.50 threshold
        threshold=0.50,
        contributions=contribs,
        is_impossible=False,
    )
    assert verdict == "CLEARED"
    assert all(r.level < 2 for r in reasons)


def test_impossible_input_enforces_level_2_and_flagged():
    """Verify that amount > sender_balance enforces FLAGGED verdict and Level 2 balance reason."""
    dummy_contribs = np.zeros(9)
    effective_risk, verdict, reasons = generate_reasons(
        tx_type=TransactionType.TRANSFER,
        amount=500000.0,
        sender_balance=50000.0,
        recipient_balance=0.0,
        time_str="14:30",
        raw_model_risk=0.0,  # Raw model returned 0
        threshold=0.50,
        contributions=dummy_contribs,
        is_impossible=True,
    )
    assert verdict == "FLAGGED"
    assert effective_risk >= 0.85
    bal_reasons = [r for r in reasons if r.signal == SignalType.BALANCE_BEHAVIOR]
    assert any(r.level == 2 for r in bal_reasons)
    assert any("Amount is larger than the sender's balance" in r.text for r in bal_reasons)
