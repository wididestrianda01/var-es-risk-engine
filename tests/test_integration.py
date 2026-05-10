import numpy as np
from src.garch import fit_garch
from src.var_methods import compute_var_es
from src.backtest import kupiec_test, christoffersen_test, acerbi_szekely_z2
from src.utils import compute_returns


def test_full_pipeline_synthetic():
    """GARCH -> VaR -> rolling backtest on synthetic data."""
    rng = np.random.default_rng(123)
    returns = rng.normal(0, 0.01, 750)

    estimation_window = 500
    var_forecasts = []
    es_forecasts = []
    realized = []

    for t in range(estimation_window, len(returns)):
        train = returns[t - estimation_window:t]
        test_return = returns[t]

        garch_result = fit_garch(train)
        result = compute_var_es(train, method="historical", alpha=0.99)

        var_forecasts.append(result.var)
        es_forecasts.append(result.es)
        realized.append(test_return)

    var_arr = np.array(var_forecasts)
    es_arr = np.array(es_forecasts)
    ret_arr = np.array(realized)
    breaches = (ret_arr <= var_arr).astype(int)

    kupiec = kupiec_test(breaches.sum(), len(breaches), alpha=0.99)
    christoffersen = christoffersen_test(breaches)
    z2 = acerbi_szekely_z2(ret_arr, var_arr, es_arr, alpha=0.99, n_sim=500)

    assert kupiec.p_value > 0.01, f"Kupiec rejected: p={kupiec.p_value}"
    assert z2.p_value > 0.01, f"Z2 rejected: p={z2.p_value}"


def test_full_pipeline_garch_to_var():
    """GARCH output feeds directly into VaR computation."""
    rng = np.random.default_rng(42)
    returns = rng.normal(0, 0.01, 500)

    garch_result = fit_garch(returns)
    result_no_garch = compute_var_es(returns, method="parametric", alpha=0.99)
    result_garch = compute_var_es(returns, method="parametric", alpha=0.99,
                                  garch_result=garch_result)

    assert result_garch.var != result_no_garch.var
    assert result_garch.garch_used is True
    assert result_no_garch.garch_used is False
