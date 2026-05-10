"""Hypothesis profiles and shared fixtures for var-es tests."""

import os
import numpy as np
import pytest
from hypothesis import strategies as st, settings, HealthCheck

# Global float strategy: never generate NaN/Inf in financial tests
st.register_type_strategy(
    float,
    st.floats(allow_nan=False, allow_infinity=False, allow_subnormal=False),
)

# Hypothesis profiles
settings.register_profile("dev", max_examples=10, deadline=None)
settings.register_profile("ci", max_examples=200, deadline=None,
                          suppress_health_check=[HealthCheck.too_slow])
settings.register_profile("nightly", max_examples=2000, deadline=None)
settings.load_profile(os.getenv("HYPOTHESIS_PROFILE", "dev"))


@pytest.fixture
def sample_returns() -> np.ndarray:
    """252 days of synthetic daily returns for tests."""
    rng = np.random.default_rng(42)
    return rng.normal(0, 0.01, 252)


@pytest.fixture
def sample_prices() -> np.ndarray:
    """252 days of synthetic prices starting at 100."""
    rng = np.random.default_rng(43)
    returns = rng.normal(0, 0.01, 251)
    prices = [100.0]
    for r in returns:
        prices.append(prices[-1] * (1 + r))
    return np.array(prices)
