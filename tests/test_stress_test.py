# tests/test_stress_test.py
import numpy as np
import pandas as pd
from src.stress_test import (StressResult, run_historical_scenario,
                              find_worst_window, sensitivity_shocks)


def test_stress_result_dataclass():
    result = StressResult(scenario="test", var=-0.05, es=-0.08,
                          pnl=-0.10, worst_day=-0.03)
    assert result.var == -0.05


def test_run_historical_scenario():
    rng = np.random.default_rng(42)
    dates = pd.date_range("2020-01-01", periods=500, freq="B")
    returns = pd.Series(rng.normal(0, 0.01, 500), index=dates)
    result = run_historical_scenario(returns, "2020-03-01", "2020-05-31",
                                     scenario_name="COVID")
    assert result.scenario == "COVID"
    assert result.var < 0
    assert result.es <= result.var
    assert result.worst_day <= 0


def test_find_worst_window():
    rng = np.random.default_rng(42)
    returns = np.zeros(500)
    returns[200:300] = rng.normal(-0.03, 0.02, 100)
    dates = pd.date_range("2015-01-01", periods=500, freq="B")
    sr = pd.Series(returns, index=dates)
    start, end = find_worst_window(sr, window_days=60)
    assert start >= pd.Timestamp("2015-10-01")
    assert end <= pd.Timestamp("2017-06-01")


def test_sensitivity_shocks():
    rng = np.random.default_rng(42)
    returns = pd.DataFrame({
        "Equity": rng.normal(0, 0.01, 500),
        "FX": rng.normal(0, 0.005, 500),
    })
    shocks = {"Equity": -0.10, "FX": 0.05}
    result = sensitivity_shocks(returns, shocks)
    assert "Equity" in result.columns
    assert "Equity_shocked" in result.columns
