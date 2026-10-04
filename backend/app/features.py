"""Feature extraction and engineering module for FRAUD SHIELD.

Transforms the 5 user-facing transaction inputs into the exact 9-feature
representation expected by the trained Random Forest model and scaler.
"""

from typing import Dict, List, Tuple
import pandas as pd
from app.schemas import PredictRequest, TransactionType

# The exact 9 features in the identical order trained in ML/train_final_model.ipynb
FEATURE_NAMES: List[str] = [
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


def parse_time_to_step(time_str: str) -> float:
    """Convert 'HH:MM' time string into fractional hours (0.00 to 23.98).

    In the PaySim dataset, 'step' maps 1 hour per increment. Mapping the 24-hour
    time of day onto this dimension allows the tree splits to capture time-of-day
    patterns (such as elevated night-time cashouts).
    """
    parts = time_str.strip().split(":")
    hour = int(parts[0])
    minute = int(parts[1]) if len(parts) > 1 else 0
    return float(hour + minute / 60.0)


def extract_features(request: PredictRequest) -> pd.DataFrame:
    """Extract and engineer the 9 model features from a validated PredictRequest.

    Feature Engineering Logic:
    1. step: Derived from the 24-hour transaction time string (HH:MM).
    2. amount: Raw transaction amount in rupees.
    3. oldbalanceOrg: Sender account balance immediately prior to transaction.
    4. newbalanceOrig: Expected sender balance post-transaction: max(0, oldbalanceOrg - amount).
    5. oldbalanceDest: Recipient account balance immediately prior to transaction.
    6. newbalanceDest:
       - For TRANSFER: Funds credit to recipient (oldbalanceDest + amount).
       - For CASH_OUT: In PaySim, cash out routes funds through an agent/mule.
         If oldbalanceDest is 0, post-transaction remains 0 (cash handed out).
         Otherwise, credits to the recipient balance.
    7. errorBalanceOrig: Accounting delta: newbalanceOrig + amount - oldbalanceOrg.
       Captures discrepancies like attempted overdrafts.
    8. errorBalanceDest: Accounting delta: oldbalanceDest + amount - newbalanceDest.
       Highlights destination sinks where funds immediately exit the ecosystem.
    9. type_TRANSFER: 1.0 if TRANSFER else 0.0 (CASH_OUT).

    Returns:
        pd.DataFrame with 1 row and columns matching FEATURE_NAMES.
    """
    step = parse_time_to_step(request.time)
    amount = float(request.amount)
    oldbalance_org = float(request.sender_balance)
    oldbalance_dest = float(request.recipient_balance)

    # Post-transaction sender balance
    newbalance_orig = max(0.0, oldbalance_org - amount)

    # Post-transaction recipient balance
    is_transfer = (request.type == TransactionType.TRANSFER)
    if is_transfer:
        newbalance_dest = oldbalance_dest + amount
    else:
        # CASH_OUT: Funds withdrawn via merchant or cashout sink
        if oldbalance_dest > 0.0:
            newbalance_dest = oldbalance_dest + amount
        else:
            newbalance_dest = 0.0

    # Engineered accounting error balances
    error_balance_orig = newbalance_orig + amount - oldbalance_org
    error_balance_dest = oldbalance_dest + amount - newbalance_dest
    type_transfer = 1.0 if is_transfer else 0.0

    feature_dict: Dict[str, float] = {
        "step": step,
        "amount": amount,
        "oldbalanceOrg": oldbalance_org,
        "newbalanceOrig": newbalance_orig,
        "oldbalanceDest": oldbalance_dest,
        "newbalanceDest": newbalance_dest,
        "errorBalanceOrig": error_balance_orig,
        "errorBalanceDest": error_balance_dest,
        "type_TRANSFER": type_transfer,
    }

    return pd.DataFrame([feature_dict], columns=FEATURE_NAMES)
