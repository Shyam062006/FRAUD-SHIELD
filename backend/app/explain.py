"""Plain-language explanation engine for FRAUD SHIELD predictions.

Translates model predictions and feature signals into clear, non-technical,
human-readable explanations grouped into the 4 standard telemetry signals:
- amount_pattern
- balance_behavior
- transaction_type
- time_pattern

No raw feature names are exposed to end users.
"""

from typing import List
import numpy as np
import pandas as pd
from app.schemas import ReasonItem, SignalType, TransactionType


def generate_reasons(
    tx_type: TransactionType,
    amount: float,
    sender_balance: float,
    recipient_balance: float,
    time_str: str,
    fraud_risk: float,
    feature_importances: np.ndarray,
) -> List[ReasonItem]:
    """Generate plain-language explanation reasons based on feature signals and model weights.

    Args:
        tx_type: TransactionType (TRANSFER or CASH_OUT)
        amount: Transaction amount in rupees
        sender_balance: Starting balance of sender
        recipient_balance: Starting balance of recipient
        time_str: HH:MM time string
        fraud_risk: Model's predicted fraud probability (0..1)
        feature_importances: Array of model feature importances

    Returns:
        List of ReasonItem objects, sorted with highest severity first.
    """
    reasons: List[ReasonItem] = []

    # 1. Parse time
    parts = time_str.split(":")
    hour = int(parts[0])

    # 2. Compute financial ratios
    depletion_ratio = amount / (sender_balance + 1.0)
    used_pct = int(min(1.0, max(0.0, depletion_ratio)) * 100)
    dest_ratio = recipient_balance / (amount + 1.0)

    # ----------------------------------------------------
    # Signal 1: Balance Behavior
    # ----------------------------------------------------
    # In the model, newbalanceOrig and errorBalanceOrig account for >60% of total importance.
    if depletion_ratio >= 0.80 or amount >= sender_balance:
        bal_level = 2
        bal_text = f"Most of the sender's balance ({used_pct}%) is being moved in one go."
    elif depletion_ratio >= 0.40:
        bal_level = 1
        bal_text = f"A large share of the sender's balance ({used_pct}%) is being moved."
    else:
        bal_level = 0
        bal_text = "The amount is small compared with the sender's balance."

    reasons.append(
        ReasonItem(
            level=bal_level,
            text=bal_text,
            signal=SignalType.BALANCE_BEHAVIOR,
        )
    )

    # Recipient balance context (secondary balance check)
    if dest_ratio < 0.10 and amount >= 3000.0:
        dest_level = 2 if amount >= 8000.0 else 1
        dest_text = "The receiving account holds very little compared with the amount sent."
        reasons.append(
            ReasonItem(
                level=dest_level,
                text=dest_text,
                signal=SignalType.BALANCE_BEHAVIOR,
            )
        )

    # ----------------------------------------------------
    # Signal 2: Time Pattern
    # ----------------------------------------------------
    if hour < 5 or hour >= 23:
        time_level = 2
        time_text = "Sent late at night or in the early hours, when fraud is more common."
    elif 5 <= hour < 7:
        time_level = 1
        time_text = "Sent early in the morning, which is slightly unusual."
    else:
        time_level = 0
        time_text = "Sent at a normal time of day."

    reasons.append(
        ReasonItem(
            level=time_level,
            text=time_text,
            signal=SignalType.TIME_PATTERN,
        )
    )

    # ----------------------------------------------------
    # Signal 3: Transaction Type
    # ----------------------------------------------------
    if tx_type == TransactionType.CASH_OUT:
        type_level = 1
        type_text = "Cash-outs move money out of the system, so they are watched more closely than transfers."
    else:
        type_level = 0
        type_text = "Direct transfer between accounts within standard clearing parameters."

    reasons.append(
        ReasonItem(
            level=type_level,
            text=type_text,
            signal=SignalType.TRANSACTION_TYPE,
        )
    )

    # ----------------------------------------------------
    # Signal 4: Amount Pattern
    # ----------------------------------------------------
    if amount >= 250000.0:
        amt_level = 2
        amt_text = "Very high-value transfer moving substantial capital in a single operation."
    elif amount >= 50000.0:
        amt_level = 1
        amt_text = "Above-average transaction value for personal account transfers."
    else:
        amt_level = 0
        amt_text = "Standard transaction amount within ordinary operational bounds."

    reasons.append(
        ReasonItem(
            level=amt_level,
            text=amt_text,
            signal=SignalType.AMOUNT_PATTERN,
        )
    )

    # Sort reasons by severity level descending (level 2 -> 1 -> 0)
    reasons.sort(key=lambda r: r.level, reverse=True)
    return reasons
