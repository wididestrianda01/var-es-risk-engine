# src/stress_test.py
from dataclasses import dataclass
import numpy as np
import pandas as pd
from src.var_methods import compute_var_es


@dataclass
class StressResult:
    scenario: str
    var: float
    es: float
    pnl: float
    worst_day: float


def run_historical_scenario(returns, start, end, scenario_name=None):
    """Run VaR/ES on a historical stress period."""
    if scenario_name is None:
        scenario_name = f"{start}-{end}"
    mask = (returns.index >= start) & (returns.index <= end)
    period_returns = returns[mask]
    if len(period_returns) < 20:
        raise ValueError(f"Period {start}-{end} has only {len(period_returns)} obs")
    var_result = compute_var_es(period_returns.values, method="historical", alpha=0.975)
    return StressResult(
        scenario=scenario_name,
        var=var_result.var,
        es=var_result.es,
        pnl=float(period_returns.sum()),
        worst_day=float(period_returns.min()),
    )


def find_worst_window(returns, window_days=252):
    """Find the period with lowest cumulative return over window_days trading days."""
    cum_returns = returns.cumsum()
    min_return = float("inf")
    worst_end_idx = None
    n = len(returns)
    for i in range(window_days, n):
        window_return = cum_returns.iloc[i] - cum_returns.iloc[i - window_days]
        if window_return < min_return:
            min_return = window_return
            worst_end_idx = i
    if worst_end_idx is None:
        return returns.index[0], returns.index[-1]
    return returns.index[worst_end_idx - window_days], returns.index[worst_end_idx]


def sensitivity_shocks(returns_df, shocks):
    """Apply sensitivity shocks to asset returns, distributing over 252 days."""
    result = returns_df.copy()
    for asset, shock in shocks.items():
        if asset in result.columns:
            result[f"{asset}_shocked"] = result[asset] + shock / 252
    return result
