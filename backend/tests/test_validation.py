"""Unit tests for Pydantic input validation limits."""

import pytest
from pydantic import ValidationError
from app.schemas import PredictRequest, TransactionType


def test_valid_payload():
    """Verify that a standard valid payload instantiates without errors."""
    req = PredictRequest(
        type=TransactionType.TRANSFER,
        amount=1200.0,
        sender_balance=5000.0,
        recipient_balance=3200.0,
        time="14:30",
    )
    assert req.type == TransactionType.TRANSFER
    assert req.amount == 1200.0


@pytest.mark.parametrize("valid_amount", [1.0, 50.0, 100000.0, 1000000.0])
def test_valid_amounts(valid_amount):
    """Verify amount boundary values [1, 1000000]."""
    req = PredictRequest(
        type=TransactionType.TRANSFER,
        amount=valid_amount,
        sender_balance=5000.0,
        recipient_balance=1000.0,
        time="12:00",
    )
    assert req.amount == valid_amount


@pytest.mark.parametrize("invalid_amount", [0.99, 0.0, -10.0, 1000000.01, 2000000.0])
def test_invalid_amounts(invalid_amount):
    """Verify that amounts outside [1, 1000000] fail validation with clear messages."""
    with pytest.raises(ValidationError) as exc:
        PredictRequest(
            type=TransactionType.TRANSFER,
            amount=invalid_amount,
            sender_balance=5000.0,
            recipient_balance=1000.0,
            time="12:00",
        )
    err_str = str(exc.value)
    assert "Amount must be at least ₹1." in err_str or "Amount cannot exceed ₹1,000,000." in err_str


@pytest.mark.parametrize("invalid_balance", [-1.0, -0.01, 10000001.0])
def test_invalid_balances(invalid_balance):
    """Verify balances must be between 0 and 10,000,000."""
    with pytest.raises(ValidationError) as exc:
        PredictRequest(
            type=TransactionType.TRANSFER,
            amount=100.0,
            sender_balance=invalid_balance,
            recipient_balance=1000.0,
            time="12:00",
        )
    assert "balance" in str(exc.value).lower()


@pytest.mark.parametrize("valid_time", ["00:00", "02:15", "14:30", "23:59"])
def test_valid_times(valid_time):
    """Verify 24-hour HH:MM format strings."""
    req = PredictRequest(
        type=TransactionType.TRANSFER,
        amount=100.0,
        sender_balance=1000.0,
        recipient_balance=1000.0,
        time=valid_time,
    )
    assert req.time == valid_time


@pytest.mark.parametrize("invalid_time", ["24:00", "12:60", "25:00", "9:00", "14:3", "abc", ""])
def test_invalid_times(invalid_time):
    """Verify invalid time strings trigger validation errors."""
    with pytest.raises(ValidationError) as exc:
        PredictRequest(
            type=TransactionType.TRANSFER,
            amount=100.0,
            sender_balance=1000.0,
            recipient_balance=1000.0,
            time=invalid_time,
        )
    assert "Time must be in 24-hour HH:MM format" in str(exc.value)


@pytest.mark.parametrize("invalid_type", ["PAYMENT", "DEBIT", "CASH_IN", "transfer", "random"])
def test_invalid_transaction_types(invalid_type):
    """Verify only TRANSFER and CASH_OUT are permitted."""
    with pytest.raises(ValidationError):
        PredictRequest(
            type=invalid_type,
            amount=100.0,
            sender_balance=1000.0,
            recipient_balance=1000.0,
            time="12:00",
        )
