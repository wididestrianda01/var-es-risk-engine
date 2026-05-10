import numpy as np
import pytest

from src.backtest import TestResult, kupiec_test


def test_kupiec_no_breaches():
    """1 breach in 250 days at 99% — should NOT reject."""
    result = kupiec_test(breaches=1, total=250, alpha=0.99)
    assert isinstance(result, TestResult)
    assert result.p_value > 0.05  # should not reject
    assert not result.reject


def test_kupiec_many_breaches():
    """20 breaches in 250 days at 99% — should reject."""
    result = kupiec_test(breaches=20, total=250, alpha=0.99)
    assert result.reject
    assert result.p_value < 0.05


def test_kupiec_exact_expected():
    """At 99%, expect 2.5 breaches per 250 days. 2 breaches should be fine."""
    result = kupiec_test(breaches=2, total=250, alpha=0.99)
    assert not result.reject


def test_kupiec_validation():
    """breaches > total should raise."""
    with pytest.raises(ValueError):
        kupiec_test(breaches=300, total=250, alpha=0.99)
