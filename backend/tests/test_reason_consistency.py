"""Permanent test suite verifying explanation consistency and model grounding.

Strict Invariants Enforced across >= 20 diverse inputs:
1. If verdict is CLEARED, NO reason may be level 2 (red).
2. If amount > sender_balance, transaction is FLAGGED with level 2 reason "Amount is larger than the sender's balance."
3. A reason may be level 2 (red) or level 1 (amber) ONLY if that signal pushes risk UP for this prediction.
4. Explanations always cover all 4 telemetry signals, sorted by descending level.
5. No generic filler phrases ("within standard clearing parameters") exist in reason texts.
"""

import pytest
from fastapi.testclient import TestClient
from app.main import app
from app.features import extract_features
from app.model_service import model_service
from app.schemas import SignalType

# 25 Diverse Test Scenarios covering edge cases, legit, fraud, overdraft, boundaries
VARIED_TEST_INPUTS = [
    # 1. Bug report scenario (Overdraft / Impossible TRANSFER)
    {"type": "TRANSFER", "amount": 500000.0, "sender_balance": 50000.0, "recipient_balance": 0.0, "time": "14:30"},
    # 2. Overdraft CASH_OUT
    {"type": "CASH_OUT", "amount": 10000.0, "sender_balance": 500.0, "recipient_balance": 0.0, "time": "10:00"},
    # 3. Overdraft small amount
    {"type": "TRANSFER", "amount": 100.0, "sender_balance": 10.0, "recipient_balance": 5000.0, "time": "12:00"},
    # 4. Standard clear legit transfer
    {"type": "TRANSFER", "amount": 250.0, "sender_balance": 5000.0, "recipient_balance": 3200.0, "time": "14:30"},
    # 5. Standard clear legit cashout
    {"type": "CASH_OUT", "amount": 500.0, "sender_balance": 20000.0, "recipient_balance": 1000.0, "time": "11:15"},
    # 6. Night-time legit small transfer
    {"type": "TRANSFER", "amount": 50.0, "sender_balance": 10000.0, "recipient_balance": 2000.0, "time": "02:30"},
    # 7. Night-time high depletion cashout (classic fraud pattern)
    {"type": "CASH_OUT", "amount": 9800.0, "sender_balance": 10000.0, "recipient_balance": 0.0, "time": "02:15"},
    # 8. Large legit transfer with wealthy sender
    {"type": "TRANSFER", "amount": 100000.0, "sender_balance": 5000000.0, "recipient_balance": 500000.0, "time": "13:00"},
    # 9. Large transfer full balance drain
    {"type": "TRANSFER", "amount": 250000.0, "sender_balance": 250000.0, "recipient_balance": 0.0, "time": "03:45"},
    # 10. Micro transaction minimum boundary
    {"type": "TRANSFER", "amount": 1.0, "sender_balance": 100.0, "recipient_balance": 50.0, "time": "00:01"},
    # 11. Large legitimate payroll transfer
    {"type": "TRANSFER", "amount": 450000.0, "sender_balance": 1000000.0, "recipient_balance": 100000.0, "time": "15:30"},
    # 12. Mid-day high ratio cash out
    {"type": "CASH_OUT", "amount": 49000.0, "sender_balance": 50000.0, "recipient_balance": 0.0, "time": "13:45"},
    # 13. Early morning transfer
    {"type": "TRANSFER", "amount": 1200.0, "sender_balance": 15000.0, "recipient_balance": 600.0, "time": "05:15"},
    # 14. Late evening cash out
    {"type": "CASH_OUT", "amount": 2500.0, "sender_balance": 3000.0, "recipient_balance": 0.0, "time": "23:45"},
    # 15. Near-overdraft (amount == balance)
    {"type": "TRANSFER", "amount": 10000.0, "sender_balance": 10000.0, "recipient_balance": 0.0, "time": "09:00"},
    # 16. Daytime modest transfer
    {"type": "TRANSFER", "amount": 3500.0, "sender_balance": 80000.0, "recipient_balance": 12000.0, "time": "16:20"},
    # 17. High balance sender, high balance recipient
    {"type": "TRANSFER", "amount": 50000.0, "sender_balance": 8000000.0, "recipient_balance": 4000000.0, "time": "10:30"},
    # 18. Massive overdraft attempt
    {"type": "TRANSFER", "amount": 900000.0, "sender_balance": 100.0, "recipient_balance": 0.0, "time": "01:00"},
    # 19. Cash out with zero recipient initial balance
    {"type": "CASH_OUT", "amount": 15000.0, "sender_balance": 20000.0, "recipient_balance": 0.0, "time": "14:00"},
    # 20. Cash out with non-zero recipient initial balance
    {"type": "CASH_OUT", "amount": 15000.0, "sender_balance": 20000.0, "recipient_balance": 50000.0, "time": "14:00"},
    # 21. Transfer at boundary maximum amount
    {"type": "TRANSFER", "amount": 1000000.0, "sender_balance": 2000000.0, "recipient_balance": 1000000.0, "time": "12:00"},
    # 22. Off-peak morning cash out
    {"type": "CASH_OUT", "amount": 8000.0, "sender_balance": 8500.0, "recipient_balance": 0.0, "time": "06:15"},
    # 23. Normal evening grocery transfer
    {"type": "TRANSFER", "amount": 150.0, "sender_balance": 12000.0, "recipient_balance": 50000.0, "time": "19:30"},
    # 24. Exact 1 rupee overdraft
    {"type": "TRANSFER", "amount": 501.0, "sender_balance": 500.0, "recipient_balance": 100.0, "time": "10:00"},
    # 25. Large transfer late night
    {"type": "TRANSFER", "amount": 300000.0, "sender_balance": 300000.0, "recipient_balance": 0.0, "time": "01:30"},
]


@pytest.fixture(scope="module")
def client():
    """Test client with lifespan initialization."""
    with TestClient(app) as test_client:
        yield test_client


def test_twenty_plus_inputs_exist():
    """Ensure at least 20 diverse test cases are included."""
    assert len(VARIED_TEST_INPUTS) >= 20


@pytest.mark.parametrize("payload", VARIED_TEST_INPUTS)
def test_reason_consistency_invariants(client, payload):
    """Enforce model consistency rules on all test inputs."""
    res = client.post("/api/predict", json=payload)
    assert res.status_code == 200
    data = res.json()

    verdict = data["verdict"]
    reasons = data["reasons"]
    amount = payload["amount"]
    sender_bal = payload["sender_balance"]
    is_impossible = amount > sender_bal

    # Rule 1: Explanations must cover all 4 telemetry signals
    signal_set = {r["signal"] for r in reasons}
    assert signal_set == {
        SignalType.BALANCE_BEHAVIOR.value,
        SignalType.TIME_PATTERN.value,
        SignalType.TRANSACTION_TYPE.value,
        SignalType.AMOUNT_PATTERN.value,
    }

    # Rule 2: Reasons must be sorted descending by level
    levels = [r["level"] for r in reasons]
    assert levels == sorted(levels, reverse=True)

    # Rule 3: If verdict is CLEARED, NO reason may be level 2 (Red)
    if verdict == "CLEARED":
        for r in reasons:
            assert r["level"] < 2, (
                f"Violation: Verdict is CLEARED but found Level 2 reason '{r['text']}' "
                f"for input {payload}"
            )

    # Rule 4: If amount > sender_balance, verdict must be FLAGGED with level 2 reason
    if is_impossible:
        assert verdict == "FLAGGED", (
            f"Violation: Impossible input {payload} was not FLAGGED."
        )
        bal_reasons = [r for r in reasons if r["signal"] == SignalType.BALANCE_BEHAVIOR.value]
        assert any(r["level"] == 2 for r in bal_reasons), (
            f"Violation: Impossible input {payload} lacks Level 2 balance reason."
        )
        assert any("Amount is larger than the sender's balance" in r["text"] for r in bal_reasons)

    # Rule 5: A reason may be level 2 or 1 ONLY if that signal pushes risk UP for this prediction
    # Verify using direct model service contributions
    from app.schemas import PredictRequest, TransactionType
    req = PredictRequest(
        type=TransactionType(payload["type"]),
        amount=payload["amount"],
        sender_balance=payload["sender_balance"],
        recipient_balance=payload["recipient_balance"],
        time=payload["time"],
    )
    feats = extract_features(req)
    _, _, contribs = model_service.predict_risk_and_contributions(feats)

    phi_time = float(contribs[0])
    phi_amount = float(contribs[1])
    phi_balance = float(contribs[[2, 3, 4, 5, 6, 7]].sum())
    phi_type = float(contribs[8])

    for r in reasons:
        lvl = r["level"]
        sig = r["signal"]
        if lvl >= 1:
            if sig == SignalType.BALANCE_BEHAVIOR.value:
                assert is_impossible or phi_balance > 0.01, (
                    f"Balance signal level {lvl} without positive contribution: {phi_balance}"
                )
            elif sig == SignalType.TIME_PATTERN.value:
                assert phi_time > 0.01, (
                    f"Time signal level {lvl} without positive contribution: {phi_time}"
                )
            elif sig == SignalType.TRANSACTION_TYPE.value:
                assert phi_type > 0.01, (
                    f"Type signal level {lvl} without positive contribution: {phi_type}"
                )
            elif sig == SignalType.AMOUNT_PATTERN.value:
                assert phi_amount > 0.01, (
                    f"Amount signal level {lvl} without positive contribution: {phi_amount}"
                )

    # Rule 6: No filler phrases in any reason text
    forbidden_fillers = ["standard clearing parameters", "standard operating threshold"]
    for r in reasons:
        for filler in forbidden_fillers:
            assert filler not in r["text"].lower(), (
                f"Filler phrase '{filler}' found in reason text: '{r['text']}'"
            )
