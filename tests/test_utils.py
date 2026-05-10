"""Tests for data fetching and return computation utilities."""

import numpy as np
import pandas as pd
import pytest

from src.utils import compute_returns


class TestComputeReturns:
    """Tests for the compute_returns function."""

    def test_compute_returns_log(self):
        prices = pd.Series([100.0, 101.0, 99.0])
        result = compute_returns(prices, method="log")
        expected = np.array([np.log(101 / 100), np.log(99 / 101)])
        assert np.allclose(result, expected)

    def test_compute_returns_simple(self):
        prices = pd.Series([100.0, 102.0, 98.0])
        result = compute_returns(prices, method="simple")
        expected = np.array([0.02, -0.0392156862745098])
        assert np.allclose(result, expected)

    def test_compute_returns_constant_prices(self):
        prices = pd.Series([100.0, 100.0, 100.0])
        result = compute_returns(prices, method="log")
        assert np.allclose(result, [0.0, 0.0])

    def test_invalid_method_raises(self):
        prices = pd.Series([100.0, 101.0])
        with pytest.raises(ValueError, match="Unknown method"):
            compute_returns(prices, method="invalid")
