"""Model-grounded explanation engine for FRAUD SHIELD predictions.

Translates instance tree contributions (Saabas/TreeInterpreter) and domain signals
into plain-language, non-technical explanations grouped into the 4 standard telemetry signals:
- balance_behavior
- time_pattern
- transaction_type
- amount_pattern

Strict Behavioral Invariants:
1. A reason may be Level 2 (red) or Level 1 (amber) ONLY if that signal pushes risk UP
   for this specific prediction (or violates the sender balance invariant).
2. If verdict is CLEARED and risk is below the threshold, NO reason may be Level 2.
3. No raw internal feature names are exposed to end users.
"""

from typing import List, Tuple
import numpy as np
from app.schemas import ReasonItem, SignalType, TransactionType


def generate_reasons(
    tx_type: TransactionType,
    amount: float,
    sender_balance: float,
    recipient_balance: float,
    time_str: str,
    raw_model_risk: float,
    threshold: float,
    contributions: np.ndarray,
    is_impossible: bool = False,
) -> Tuple[float, str, List[ReasonItem]]:
    """Generate plain-language explanation reasons directly grounded in tree path contributions.

    Args:
        tx_type: TransactionType (TRANSFER or CASH_OUT)
        amount: Transaction amount in rupees
        sender_balance: Starting balance of sender
        recipient_balance: Starting balance of recipient
        time_str: HH:MM time string
        raw_model_risk: Pure model predicted fraud probability (0..1)
        threshold: Classification decision threshold (e.g. 0.50)
        contributions: Array of 9 feature contributions for this instance
        is_impossible: True if amount > sender_balance

    Returns:
        Tuple of (effective_risk: float, verdict: str, reasons: List[ReasonItem])
    """
    # 1. Determine effective risk and verdict
    if is_impossible:
        effective_risk = max(raw_model_risk, 0.85)
        verdict = "FLAGGED"
    else:
        effective_risk = raw_model_risk
        verdict = "FLAGGED" if effective_risk >= threshold else "CLEARED"

    is_flagged = (verdict == "FLAGGED")

    # 2. Decompose contributions across the 4 signal categories:
    # 0: step, 1: amount, 2: oldbalanceOrg, 3: newbalanceOrig, 4: oldbalanceDest,
    # 5: newbalanceDest, 6: errorBalanceOrig, 7: errorBalanceDest, 8: type_TRANSFER
    phi_time = float(contributions[0])
    phi_amount = float(contributions[1])
    phi_balance = float(contributions[[2, 3, 4, 5, 6, 7]].sum())
    phi_type = float(contributions[8])

    # 3. Domain financial ratios and context
    depletion_ratio = amount / (sender_balance + 1.0)
    used_pct = int(min(1.0, max(0.0, depletion_ratio)) * 100)
    dest_ratio = recipient_balance / (amount + 1.0)
    hour = int(time_str.split(":")[0])

    reasons: List[ReasonItem] = []

    # -------------------------------------------------------------------------
    # Signal 1: Balance Behavior
    # -------------------------------------------------------------------------
    if is_impossible:
        bal_level = 2
        bal_text = "Amount is larger than the sender's balance."
    elif phi_balance > 0.01:
        if is_flagged and (phi_balance >= 0.08 or depletion_ratio >= 0.80):
            bal_level = 2
            if dest_ratio < 0.10 and amount >= 3000.0:
                bal_text = f"Most of the sender's balance ({used_pct}%) is being moved in one go to an empty receiving account."
            else:
                bal_text = f"Most of the sender's balance ({used_pct}%) is being moved in one go."
        else:
            bal_level = 1
            bal_text = f"A large share of the sender's balance ({used_pct}%) is being moved."
    else:
        bal_level = 0
        bal_text = "The transaction amount is well within the sender's available balance."

    reasons.append(
        ReasonItem(
            level=bal_level,
            text=bal_text,
            signal=SignalType.BALANCE_BEHAVIOR,
        )
    )

    # -------------------------------------------------------------------------
    # Signal 2: Time Pattern
    # -------------------------------------------------------------------------
    if phi_time > 0.01 and (hour < 7 or hour >= 23):
        if is_flagged and (phi_time >= 0.08 or hour < 5 or hour >= 23):
            time_level = 2
            time_text = "Sent late at night or in the early morning hours, when fraud rates peak."
        else:
            time_level = 1
            time_text = f"Sent during off-peak morning hours ({time_str}), slightly increasing risk."
    else:
        time_level = 0
        time_text = f"Sent at a normal operating hour ({time_str})."

    reasons.append(
        ReasonItem(
            level=time_level,
            text=time_text,
            signal=SignalType.TIME_PATTERN,
        )
    )

    # -------------------------------------------------------------------------
    # Signal 3: Transaction Type
    # -------------------------------------------------------------------------
    if phi_type > 0.01:
        if tx_type == TransactionType.CASH_OUT:
            if is_flagged and phi_type >= 0.08:
                type_level = 2
                type_text = "Cash-outs immediately extract liquidity outside the monitored banking network."
            else:
                type_level = 1
                type_text = "Cash-outs move money out of the banking network, increasing scrutiny."
        else:
            if is_flagged and phi_type >= 0.08:
                type_level = 2
                type_text = "High-velocity account transfer frequently used to stage fraudulent funds."
            else:
                type_level = 1
                type_text = "Transfer channel exhibits higher baseline risk than standard operations."
    else:
        type_level = 0
        if tx_type == TransactionType.CASH_OUT:
            type_text = "Cash-out transaction within normal account profile."
        else:
            type_text = "Direct transfer between registered accounts."

    reasons.append(
        ReasonItem(
            level=type_level,
            text=type_text,
            signal=SignalType.TRANSACTION_TYPE,
        )
    )

    # -------------------------------------------------------------------------
    # Signal 4: Amount Pattern
    # -------------------------------------------------------------------------
    if phi_amount > 0.01 and amount >= 50000.0:
        if is_flagged and (phi_amount >= 0.08 or amount >= 250000.0):
            amt_level = 2
            amt_text = "High-value transfer moving substantial capital in a single operation."
        else:
            amt_level = 1
            amt_text = "Above-average transaction value."
    else:
        amt_level = 0
        amt_text = "Transaction amount is within typical operating bounds."

    reasons.append(
        ReasonItem(
            level=amt_level,
            text=amt_text,
            signal=SignalType.AMOUNT_PATTERN,
        )
    )

    # -------------------------------------------------------------------------
    # Strict Consistency Invariants:
    # 1. If verdict is CLEARED, NO reason may be level 2 (Rule 2).
    # 2. A reason may be level 2 or 1 ONLY if that signal pushes risk UP (Rule 1).
    # -------------------------------------------------------------------------
    if not is_flagged:
        for r in reasons:
            if r.level == 2:
                r.level = 1

    # Sort descending by severity (level 2 -> 1 -> 0)
    reasons.sort(key=lambda r: r.level, reverse=True)
    return effective_risk, verdict, reasons
