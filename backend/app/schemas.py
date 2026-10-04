"""Pydantic schemas and data validation models for the FRAUD SHIELD API."""

from enum import Enum
import re
from typing import Any, Dict, List, Literal, Union
from pydantic import BaseModel, ConfigDict, Field, field_validator


class TransactionType(str, Enum):
    TRANSFER = "TRANSFER"
    CASH_OUT = "CASH_OUT"


class SignalType(str, Enum):
    AMOUNT_PATTERN = "amount_pattern"
    BALANCE_BEHAVIOR = "balance_behavior"
    TRANSACTION_TYPE = "transaction_type"
    TIME_PATTERN = "time_pattern"


class PredictRequest(BaseModel):
    """Transaction evaluation request schema."""

    type: TransactionType = Field(
        ...,
        description="Transaction type. Must be either 'TRANSFER' or 'CASH_OUT'.",
        examples=["TRANSFER", "CASH_OUT"],
    )
    amount: float = Field(
        ...,
        description="Transaction amount in rupees (₹1 to ₹1,000,000).",
        examples=[1200.0],
    )
    sender_balance: float = Field(
        ...,
        description="Sender starting balance in rupees (₹0 to ₹10,000,000).",
        examples=[5000.0],
    )
    recipient_balance: float = Field(
        ...,
        description="Recipient starting balance in rupees (₹0 to ₹10,000,000).",
        examples=[3200.0],
    )
    time: str = Field(
        ...,
        description="Time of day in HH:MM 24-hour format.",
        examples=["14:30"],
    )

    @field_validator("amount")
    @classmethod
    def validate_amount(cls, v: float) -> float:
        if v < 1.0:
            raise ValueError("Amount must be at least ₹1.")
        if v > 1000000.0:
            raise ValueError("Amount cannot exceed ₹1,000,000.")
        return float(v)

    @field_validator("sender_balance")
    @classmethod
    def validate_sender_balance(cls, v: float) -> float:
        if v < 0.0:
            raise ValueError("Sender balance cannot be negative.")
        if v > 10000000.0:
            raise ValueError("Sender balance cannot exceed ₹10,000,000.")
        return float(v)

    @field_validator("recipient_balance")
    @classmethod
    def validate_recipient_balance(cls, v: float) -> float:
        if v < 0.0:
            raise ValueError("Recipient balance cannot be negative.")
        if v > 10000000.0:
            raise ValueError("Recipient balance cannot exceed ₹10,000,000.")
        return float(v)

    @field_validator("time")
    @classmethod
    def validate_time(cls, v: str) -> str:
        v = v.strip()
        time_regex = r"^([01]\d|2[0-3]):([0-5]\d)$"
        if not re.match(time_regex, v):
            raise ValueError("Time must be in 24-hour HH:MM format (e.g., '14:30' or '02:15').")
        return v


class ReasonItem(BaseModel):
    """Individual explanation factor explaining risk prediction."""

    level: int = Field(
        ...,
        ge=0,
        le=2,
        description="Severity level: 0 = normal, 1 = mild warning, 2 = strong warning.",
    )
    text: str = Field(
        ...,
        description="Plain-language non-technical explanation sentence.",
    )
    signal: SignalType = Field(
        ...,
        description="Signal category: amount_pattern, balance_behavior, transaction_type, or time_pattern.",
    )


class PredictResponse(BaseModel):
    """Transaction risk evaluation response."""

    model_config = ConfigDict(protected_namespaces=())

    risk: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Model predicted probability of fraud (0.0 to 1.0).",
    )
    verdict: Literal["FLAGGED", "CLEARED"] = Field(
        ...,
        description="Decision outcome based on decision threshold.",
    )
    threshold: float = Field(
        ...,
        description="Classification decision threshold used for the verdict.",
    )
    reasons: List[ReasonItem] = Field(
        ...,
        description="List of plain-language contributing signals.",
    )
    model: str = Field(
        ...,
        description="Name of the machine learning model serving predictions.",
    )
    latency_ms: float = Field(
        ...,
        description="Inference pipeline latency in milliseconds.",
    )


class HealthResponse(BaseModel):
    """Health check response schema."""

    model_config = ConfigDict(protected_namespaces=())

    status: str
    model_loaded: bool
    model_name: str
    version: str
    uptime_seconds: float


class ModelInfoResponse(BaseModel):
    """Detailed model metadata and benchmark metrics."""

    model_config = ConfigDict(protected_namespaces=())

    model_name: str
    display_name: str
    feature_count: int
    features: List[str]
    decision_threshold: float
    production_metrics: Dict[str, float]
    benchmark_comparison: List[Dict[str, Any]]
