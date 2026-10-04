"""Unit tests for classification threshold logic."""

import pytest
from app.config import settings


def test_default_threshold_value():
    """Verify default decision threshold matches notebook benchmark (0.50)."""
    assert settings.DECISION_THRESHOLD == 0.50


@pytest.mark.parametrize(
    "risk, threshold, expected_verdict",
    [
        (0.00, 0.50, "CLEARED"),
        (0.499, 0.50, "CLEARED"),
        (0.50, 0.50, "FLAGGED"),
        (0.501, 0.50, "FLAGGED"),
        (0.99, 0.50, "FLAGGED"),
        (0.54, 0.55, "CLEARED"),
        (0.55, 0.55, "FLAGGED"),
        (0.32, 0.30, "FLAGGED"),
        (0.32, 0.35, "CLEARED"),
    ],
)
def test_verdict_determination(risk, threshold, expected_verdict):
    """Verify that risk compared to threshold produces correct FLAGGED/CLEARED outcome."""
    verdict = "FLAGGED" if risk >= threshold else "CLEARED"
    assert verdict == expected_verdict
