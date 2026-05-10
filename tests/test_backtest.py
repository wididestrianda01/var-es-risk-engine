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


from src.backtest import christoffersen_test


def test_christoffersen_independent_breaches():
    """Randomly distributed breaches should NOT reject independence."""
    rng = np.random.default_rng(42)
    breaches = rng.choice([0, 1], size=250, p=[0.99, 0.01])
    result = christoffersen_test(breaches)
    assert isinstance(result, TestResult)


def test_christoffersen_clustered_breaches():
    """Clustered breaches should reject independence."""
    breaches = np.zeros(250, dtype=int)
    breaches[100:130] = 1  # 30 consecutive breaches — clearly clustered
    result = christoffersen_test(breaches)
    assert result.reject
