"""Plot helper functions for the VaR/ES dashboard."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import plotly.graph_objects as go
from scipy import stats as sp_stats

from src.var_methods import VaRResult
from src.garch import GarchResult
from src.stress_test import StressResult

if TYPE_CHECKING:
    import pandas as pd


def plot_distribution_overlay(
    returns: np.ndarray,
    result: VaRResult,
    alpha: float,
) -> go.Figure:
    """Normal PDF vs Student-t PDF vs empirical histogram with VaR line."""
    fig = go.Figure()
    fig.add_trace(
        go.Histogram(
            x=returns,
            nbinsx=60,
            histnorm="probability density",
            name="Empirical",
            marker_color="lightblue",
            opacity=0.6,
        )
    )
    x_range = np.linspace(returns.min(), returns.max(), 200)
    mu, sigma = np.mean(returns), np.std(returns, ddof=1)
    fig.add_trace(
        go.Scatter(
            x=x_range,
            y=sp_stats.norm.pdf(x_range, mu, sigma),
            name="Normal Fit",
            line=dict(color="orange", dash="dash"),
        )
    )
    df_t, loc_t, scale_t = sp_stats.t.fit(returns)
    fig.add_trace(
        go.Scatter(
            x=x_range,
            y=sp_stats.t.pdf(x_range, df_t, loc_t, scale_t),
            name=f"Student-t Fit (ν={df_t:.1f})",
            line=dict(color="green", dash="dot"),
        )
    )
    fig.add_vline(
        x=result.var,
        line_dash="dash",
        line_color="red",
        annotation_text=f"VaR {alpha:.1%}",
    )
    fig.update_layout(
        title="Return Distribution: Empirical vs Theoretical Fits",
        xaxis_title="Log Return",
        yaxis_title="Density",
        bargap=0.05,
    )
    return fig


def plot_conditional_vol(
    garch_result: GarchResult,
    ticker_name: str,
    date_range: tuple | None = None,
) -> go.Figure:
    """Conditional volatility time series with crisis annotations.

    Adds COVID-19 shading only when the data window overlaps Feb-Mar 2020.
    """
    n = len(garch_result.cond_vol)
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            y=garch_result.cond_vol,
            name="Conditional Volatility",
            line=dict(color="steelblue", width=1.5),
        )
    )

    covid_start_idx = covid_end_idx = None
    if date_range is not None:
        import pandas as pd
        start = pd.Timestamp(date_range[0])
        end = pd.Timestamp(date_range[1])
        total_days = (end - start).days
        covid_start = pd.Timestamp("2020-02-19")
        covid_end = pd.Timestamp("2020-03-23")
        if start <= covid_end and end >= covid_start:
            frac_start = max(0, (covid_start - start).days / max(total_days, 1))
            frac_end = min(1, (covid_end - start).days / max(total_days, 1))
            covid_start_idx = int(frac_start * n)
            covid_end_idx = int(frac_end * n)

    if covid_start_idx is not None and covid_end_idx is not None:
        fig.add_vrect(
            x0=covid_start_idx,
            x1=covid_end_idx,
            fillcolor="red",
            opacity=0.1,
            annotation_text="COVID-19 Crash",
        )

    fig.update_layout(
        title=f"GARCH Conditional Volatility — {ticker_name}",
        xaxis_title="Trading Day",
        yaxis_title="Daily Volatility",
    )
    return fig


def plot_qq_residuals(
    std_residuals: np.ndarray,
    dist_name: str,
    ticker_name: str,
) -> go.Figure:
    """QQ plot of standardized residuals against Normal theoretical quantiles."""
    n = len(std_residuals)
    theoretical = sp_stats.norm.ppf((np.arange(1, n + 1) - 0.5) / n)
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=theoretical,
            y=np.sort(std_residuals),
            mode="markers",
            name="Residuals",
            marker=dict(color="steelblue", size=4, opacity=0.5),
        )
    )
    lo, hi = theoretical.min(), theoretical.max()
    fig.add_trace(
        go.Scatter(
            x=[lo, hi],
            y=[lo, hi],
            name="Normal Reference",
            line=dict(color="red", dash="dash"),
        )
    )
    fig.update_layout(
        title=f"QQ Plot of Standardized Residuals — {ticker_name} ({dist_name})",
        xaxis_title="Theoretical Quantiles (Normal)",
        yaxis_title="Sample Quantiles",
    )
    return fig


def plot_breach_timeline(
    ret_arr: np.ndarray,
    breaches_99: np.ndarray,
    breaches_975: np.ndarray,
) -> go.Figure:
    """Breach timeline with dual-condition overlay (FRTB-style)."""
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            y=ret_arr,
            name="Returns",
            line=dict(color="gray", width=0.5),
        )
    )
    idx_99 = np.where(breaches_99)[0]
    idx_975 = np.where(breaches_975)[0]
    fig.add_trace(
        go.Scatter(
            x=idx_99,
            y=ret_arr[idx_99],
            mode="markers",
            name="99% VaR Breaches",
            marker=dict(color="red", size=8, symbol="x"),
        )
    )
    idx_975_only = np.setdiff1d(idx_975, idx_99)
    fig.add_trace(
        go.Scatter(
            x=idx_975_only,
            y=ret_arr[idx_975_only],
            mode="markers",
            name="97.5% VaR Breaches",
            marker=dict(color="orange", size=6, symbol="triangle-up"),
        )
    )
    fig.update_layout(
        title="FRTB Dual-Condition Breach Map (99% + 97.5%)",
        xaxis_title="Forecast Day",
        yaxis_title="Return",
    )
    return fig


def plot_scenario_waterfall(
    scenarios_dict: dict[str, StressResult],
    baseline_var: float,
) -> go.Figure:
    """Waterfall chart: baseline VaR → crisis scenarios → worst window."""
    names = ["Baseline"]
    values = [abs(baseline_var)]
    for label, r in scenarios_dict.items():
        names.append(label)
        values.append(abs(r.var))
    fig = go.Figure(
        go.Waterfall(
            name="VaR Magnitude",
            orientation="v",
            measure=["absolute"] + ["relative"] * len(scenarios_dict),
            x=names,
            y=values,
            connector=dict(line=dict(color="gray", dash="dot")),
            increasing=dict(marker_color="darkred"),
            decreasing=dict(marker_color="steelblue"),
            totals=dict(marker_color="maroon"),
        )
    )
    fig.update_layout(
        title="Stress Scenario Waterfall: VaR Magnitude Escalation",
        yaxis_title="|VaR| (Daily Loss)",
    )
    return fig
