import numpy as np
import pytest
from src.var_methods import compute_var_es, VaRResult


def test_historical_var_es_basic():
    rng = np.random.default_rng(42)
    returns = rng.normal(0, 0.01, 500)
    result = compute_var_es(returns, method="historical", alpha=0.975)
    assert isinstance(result, VaRResult)
    assert result.var < 0  # VaR is a loss, should be negative
    assert result.es < 0
    assert result.es <= result.var  # ES >= VaR in magnitude (more negative)
    assert result.method == "historical"
    assert result.alpha == 0.975
    assert result.horizon == 1


def test_historical_var_matches_quantile():
    """Historical VaR at 95% = 5th percentile of returns."""
    returns = np.array([-0.05, -0.03, -0.01, 0.0, 0.01, 0.02, 0.04])
    result = compute_var_es(returns, method="historical", alpha=0.95)
    expected_var = np.percentile(returns, 5)
    assert np.isclose(result.var, expected_var)


def test_historical_var_different_alphas():
    rng = np.random.default_rng(42)
    returns = rng.normal(0, 0.01, 500)
    var95 = compute_var_es(returns, method="historical", alpha=0.95)
    var99 = compute_var_es(returns, method="historical", alpha=0.99)
    # Higher confidence = larger loss (more negative VaR)
    assert var99.var <= var95.var



def test_compute_var_rejects_invalid_alpha():
    returns = np.random.randn(100)
    with pytest.raises(ValueError):
        compute_var_es(returns, method="historical", alpha=1.5)
    with pytest.raises(ValueError):
        compute_var_es(returns, method="historical", alpha=-0.1)


def test_compute_var_rejects_invalid_method():
    returns = np.random.randn(100)
    with pytest.raises(ValueError):
        compute_var_es(returns, method="crystal_ball", alpha=0.95)


def test_compute_var_rejects_invalid_horizon():
    returns = np.random.randn(100)
    with pytest.raises(ValueError):
        compute_var_es(returns, method="historical", alpha=0.95, horizon=0)


def test_compute_var_rejects_short_returns():
    with pytest.raises(ValueError):
        compute_var_es(np.array([0.01]), method="historical", alpha=0.95)


def test_parametric_var_normal():
    rng = np.random.default_rng(42)
    returns = rng.normal(0, 0.01, 500)
    result = compute_var_es(returns, method="parametric", alpha=0.975)
    assert result.var < 0
    assert result.es < 0
    assert result.es <= result.var
    assert result.method == "parametric"


def test_parametric_var_t_dist():
    """With t-dist, VaR should be more extreme than Normal for same data."""
    from scipy import stats as sp_stats
    rng = np.random.default_rng(42)
    returns = sp_stats.t.rvs(df=3, size=500) * 0.01  # fat tails
    # Use high alpha (99.5%) so t-distribution's heavier tails dominate
    # over the scale contraction from MLE fitting.
    result_norm = compute_var_es(returns, method="parametric", alpha=0.995, dist="normal")
    result_t = compute_var_es(returns, method="parametric", alpha=0.995, dist="t")
    # t-distribution VaR should be more negative (fatter tails)
    assert result_t.var < result_norm.var
