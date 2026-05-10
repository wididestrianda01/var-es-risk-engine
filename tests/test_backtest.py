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


from src.backtest import traffic_light


def test_traffic_light_basel1996_green():
    result = traffic_light(breaches=3, total=250, framework="basel1996")
    assert result["zone"] == "green"
    assert result["multiplier"] == 3.0


def test_traffic_light_basel1996_yellow():
    result = traffic_light(breaches=7, total=250, framework="basel1996")
    assert result["zone"] == "yellow"
    assert 3.4 <= result["multiplier"] <= 3.85


def test_traffic_light_basel1996_red():
    result = traffic_light(breaches=12, total=250, framework="basel1996")
    assert result["zone"] == "red"
    assert result["multiplier"] == 4.0


def test_traffic_light_frtb_green():
    result = traffic_light(breaches_99=10, breaches_975=25, total=250, framework="frtb2019")
    assert result["zone"] == "green"


def test_traffic_light_frtb_red_99():
    """>12 breaches at 99% -> red even if 97.5% is fine."""
    result = traffic_light(breaches_99=15, breaches_975=5, total=250, framework="frtb2019")
    assert result["zone"] == "red"


def test_traffic_light_frtb_red_975():
    """>30 breaches at 97.5% -> red even if 99% is fine."""
    result = traffic_light(breaches_99=5, breaches_975=35, total=250, framework="frtb2019")
    assert result["zone"] == "red"


from src.backtest import acerbi_szekely_z2


def test_acerbi_szekely_z2_correct_model():
    """When ES forecasts are correct, Z₂ should not reject."""
    rng = np.random.default_rng(42)
    returns = rng.normal(0, 0.01, 500)
    var_forecasts = np.full(500, np.percentile(returns, 5))
    es_forecasts = np.full(500, returns[returns <= var_forecasts[0]].mean())
    result = acerbi_szekely_z2(returns, var_forecasts, es_forecasts, alpha=0.95, n_sim=200)
    assert result.p_value >= 0.01


def test_acerbi_szekely_z2_bad_model():
    """When ES forecasts are too optimistic, Z₂ should reject."""
    rng = np.random.default_rng(42)
    returns = rng.normal(0, 0.01, 500)
    var_forecasts = np.full(500, np.percentile(returns, 5))
    es_forecasts = np.full(500, -0.005)  # way too optimistic
    result = acerbi_szekely_z2(returns, var_forecasts, es_forecasts, alpha=0.95, n_sim=200)
    assert result.reject
