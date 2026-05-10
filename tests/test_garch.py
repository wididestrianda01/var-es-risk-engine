"""Tests for GARCH volatility modeling."""

import numpy as np
import pytest
from hypothesis import given, strategies as st

from src.garch import fit_garch, GarchResult


class TestFitGarch:
    """Tests for the fit_garch function."""

    def test_fit_garch_returns_garch_result(self):
        rng = np.random.default_rng(42)
        returns = rng.normal(0, 0.01, 500)
        result = fit_garch(returns)
        assert isinstance(result, GarchResult)
        assert len(result.cond_vol) == len(returns)
        assert result.cond_vol.min() > 0
        assert "omega" in result.params
        assert "alpha[1]" in result.params
        assert "beta[1]" in result.params

    def test_fit_garch_rejects_short_series(self):
        rng = np.random.default_rng(42)
        returns = rng.normal(0, 0.01, 50)
        with pytest.raises(ValueError, match="at least 100"):
            fit_garch(returns)


@given(
    st.lists(
        st.floats(min_value=-0.1, max_value=0.1), min_size=100, max_size=300
    ).filter(lambda xs: np.std(xs) > 1e-12)
)
def test_garch_vol_always_positive(returns):
    """Conditional volatility must always be positive for non-degenerate series."""
    result = fit_garch(np.array(returns))
    assert np.all(result.cond_vol > 0)
