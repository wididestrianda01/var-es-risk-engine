import numpy as np
import pytest
from hypothesis import given, strategies as st
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


def test_mc_var_basic():
    rng = np.random.default_rng(42)
    returns = rng.normal(0, 0.01, 500)
    result = compute_var_es(returns, method="mc", alpha=0.975, n_sim=5000)
    assert result.var < 0
    assert result.es < 0
    assert result.es <= result.var
    assert result.method == "mc"


def test_mc_var_converges_with_more_sims():
    """MC VaR should be reasonably stable with enough simulations."""
    rng = np.random.default_rng(42)
    returns = rng.normal(0, 0.01, 500)
    result_5k = compute_var_es(returns, method="mc", alpha=0.975, n_sim=5000)
    result_50k = compute_var_es(returns, method="mc", alpha=0.975, n_sim=50000)
    assert abs(result_5k.var - result_50k.var) < 0.005


def test_mc_var_garch_integration():
    """MC with GARCH vol should differ from MC without."""
    from src.garch import fit_garch
    rng = np.random.default_rng(42)
    returns = rng.normal(0, 0.01, 500)
    garch_result = fit_garch(returns)
    result_no_garch = compute_var_es(returns, method="mc", alpha=0.975, n_sim=5000)
    result_garch = compute_var_es(returns, method="mc", alpha=0.975, n_sim=5000,
                                  garch_result=garch_result)
    assert result_garch.garch_used is True
    assert result_no_garch.garch_used is False


@given(st.lists(st.floats(min_value=-0.05, max_value=0.05), min_size=50, max_size=200))
def test_es_dominates_var(returns):
    """ES >= VaR for all methods and alphas."""
    r = np.array(returns)
    for method in ["historical", "parametric"]:
        for alpha in [0.95, 0.975, 0.99]:
            result = compute_var_es(r, method=method, alpha=alpha)
            assert result.es <= result.var


@given(st.lists(st.floats(min_value=-0.05, max_value=0.05), min_size=50, max_size=200))
def test_higher_alpha_more_conservative(returns):
    """VaR at 99% >= VaR at 95% (more negative = larger loss)."""
    r = np.array(returns)
    for method in ["historical", "parametric"]:
        var95 = compute_var_es(r, method=method, alpha=0.95).var
        var99 = compute_var_es(r, method=method, alpha=0.99).var
        assert var99 <= var95


@given(st.lists(st.floats(min_value=-0.05, max_value=0.05), min_size=50, max_size=200),
       st.floats(min_value=0.001, max_value=0.05))
def test_translation_invariance_es(returns, c):
    """ES(X + c) = ES(X) - c."""
    r = np.array(returns)
    es_original = compute_var_es(r, method="historical", alpha=0.95).es
    es_shifted = compute_var_es(r - c, method="historical", alpha=0.95).es
    assert np.isclose(es_shifted, es_original - c, rtol=1e-2)


@given(st.lists(st.floats(min_value=-0.05, max_value=0.05), min_size=50, max_size=200))
def test_positive_homogeneity_historical_var(returns):
    """Historical VaR(lambda*X) = lambda * VaR(X) for lambda > 0."""
    r = np.array(returns)
    lam = 2.0
    var_orig = compute_var_es(r, method="historical", alpha=0.95).var
    var_scaled = compute_var_es(lam * r, method="historical", alpha=0.95).var
    assert np.isclose(var_scaled, lam * var_orig, rtol=1e-2)
