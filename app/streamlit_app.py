"""VaR & Expected Shortfall Risk Engine — Streamlit Dashboard."""

import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from src.backtest import (acerbi_szekely_z2, christoffersen_test, kupiec_test,
                          traffic_light)
from src.garch import fit_garch
from src.stress_test import find_worst_window, run_historical_scenario
from src.utils import compute_returns, fetch_prices
from src.var_methods import compute_var_es

# ── Constants: Swedish equity universe (aligned with notebooks) ──────────

ASSETS = ["^OMX", "ERIC-B.ST", "VOLV-B.ST", "HM-B.ST", "SWED-A.ST"]
NAMES  = ["OMXS30", "Ericsson", "Volvo", "H&M", "Swedbank"]
SECTORS = ["Broad Market Index", "Telecom Equipment", "Industrial / Automotive",
           "Consumer Retail", "Financial / Banking"]
NAME_MAP = dict(zip(ASSETS, NAMES))
SECTOR_MAP = dict(zip(ASSETS, SECTORS))
ALPHAS = [0.95, 0.975, 0.99]

st.set_page_config(page_title="VaR & ES Engine", layout="wide")
st.title("VaR & Expected Shortfall Risk Engine")
st.caption("FRTB-aligned | Basel III | Multi-asset risk analytics")

# ── Sidebar ──────────────────────────────────────────────────────────────────

with st.sidebar:
    st.header("Controls")
    ticker = st.selectbox(
        "Asset",
        ASSETS,
        format_func=lambda x: f"{NAME_MAP[x]} ({x})  — {SECTOR_MAP[x]}"
    )
    alpha = st.selectbox(
        "Confidence Level",
        ALPHAS,
        index=1,
        format_func=lambda x: f"{x * 100:.1f}%",
    )
    horizon = st.number_input("Horizon (days)", 1, 30, 1)
    method = st.selectbox("Method", ["historical", "parametric", "mc"])
    use_garch = st.checkbox("Use GARCH volatility", value=True)
    date_range = st.date_input(
        "Date Range",
        value=(pd.Timestamp("2020-01-01"), pd.Timestamp("2025-12-31")),
    )


@st.cache_data(ttl=3600)
def load_data(ticker, start, end):
    prices = fetch_prices(ticker, str(start), str(end))
    returns = compute_returns(prices, method="log")
    return prices, returns


# ── Plot Helper Functions ──────────────────────────────────────────────────

def _plot_distribution_overlay(returns, result, alpha):
    """Normal PDF vs Student-t PDF vs empirical histogram with VaR line."""
    from scipy import stats as sp_stats
    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=returns, nbinsx=60, histnorm="probability density",
        name="Empirical", marker_color="lightblue", opacity=0.6
    ))
    x_range = np.linspace(returns.min(), returns.max(), 200)
    mu, sigma = np.mean(returns), np.std(returns, ddof=1)
    fig.add_trace(go.Scatter(
        x=x_range, y=sp_stats.norm.pdf(x_range, mu, sigma),
        name="Normal Fit", line=dict(color="orange", dash="dash")
    ))
    df_t, loc_t, scale_t = sp_stats.t.fit(returns)
    fig.add_trace(go.Scatter(
        x=x_range, y=sp_stats.t.pdf(x_range, df_t, loc_t, scale_t),
        name=f"Student-t Fit (ν={df_t:.1f})",
        line=dict(color="green", dash="dot")
    ))
    fig.add_vline(x=result.var, line_dash="dash", line_color="red",
                  annotation_text=f"VaR {alpha:.1%}")
    fig.update_layout(
        title="Return Distribution: Empirical vs Theoretical Fits",
        xaxis_title="Log Return", yaxis_title="Density",
        bargap=0.05
    )
    return fig


def _plot_conditional_vol(garch_result, ticker_name):
    """Conditional volatility time series with crisis annotations."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        y=garch_result.cond_vol,
        name="Conditional Volatility",
        line=dict(color="steelblue", width=1.5)
    ))
    # COVID-19 shaded region (approximate position — assumes 2010-2025 data, ~250 days/yr)
    n = len(garch_result.cond_vol)
    fig.add_vrect(
        x0=n - 1300, x1=n - 1220,
        fillcolor="red", opacity=0.1,
        annotation_text="COVID-19 Crash"
    )
    fig.update_layout(
        title=f"GARCH Conditional Volatility — {ticker_name}",
        xaxis_title="Date", yaxis_title="Daily Volatility"
    )
    return fig


def _plot_qq_residuals(std_residuals, dist_name, ticker_name):
    """QQ plot of standardized residuals against theoretical distribution."""
    from scipy import stats as sp_stats
    fig = go.Figure()
    n = len(std_residuals)
    theoretical = sp_stats.norm.ppf((np.arange(1, n + 1) - 0.5) / n)
    fig.add_trace(go.Scatter(
        x=theoretical, y=np.sort(std_residuals),
        mode="markers", name="Residuals",
        marker=dict(color="steelblue", size=4, opacity=0.5)
    ))
    lo, hi = theoretical.min(), theoretical.max()
    fig.add_trace(go.Scatter(
        x=[lo, hi], y=[lo, hi],
        name="Normal Reference", line=dict(color="red", dash="dash")
    ))
    fig.update_layout(
        title=f"QQ Plot of Standardized Residuals — {ticker_name} ({dist_name})",
        xaxis_title="Theoretical Quantiles (Normal)",
        yaxis_title="Sample Quantiles"
    )
    return fig


def _plot_half_life_comparison(half_lives, asset_names):
    """Horizontal bar chart comparing volatility half-life across assets."""
    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=asset_names, x=half_lives, orientation="h",
        marker_color="steelblue", text=[f"{h:.0f} days" for h in half_lives],
        textposition="outside"
    ))
    fig.update_layout(
        title="Volatility Half-Life Comparison",
        xaxis_title="Half-Life (Trading Days)",
        yaxis_title="",
        xaxis=dict(range=[0, max(half_lives) * 1.3])
    )
    return fig


def _plot_breach_timeline(ret_arr, breaches_99, breaches_975):
    """Breach timeline with dual-condition overlay (FRTB-style)."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        y=ret_arr, name="Returns", line=dict(color="gray", width=0.5)
    ))
    idx_99 = np.where(breaches_99)[0]
    idx_975 = np.where(breaches_975)[0]
    fig.add_trace(go.Scatter(
        x=idx_99, y=ret_arr[idx_99],
        mode="markers", name="99% VaR Breaches",
        marker=dict(color="red", size=8, symbol="x")
    ))
    idx_975_only = np.setdiff1d(idx_975, idx_99)
    fig.add_trace(go.Scatter(
        x=idx_975_only, y=ret_arr[idx_975_only],
        mode="markers", name="97.5% VaR Breaches",
        marker=dict(color="orange", size=6, symbol="triangle-up")
    ))
    fig.update_layout(
        title="FRTB Dual-Condition Breach Map (99% + 97.5%)",
        xaxis_title="Forecast Day", yaxis_title="Return"
    )
    return fig


def _plot_scenario_waterfall(scenarios_dict, baseline_var):
    """Waterfall chart: baseline VaR → crisis scenarios → worst window."""
    names = ["Baseline"]
    values = [abs(baseline_var)]
    for label, r in scenarios_dict.items():
        names.append(label)
        values.append(abs(r.var))
    fig = go.Figure(go.Waterfall(
        name="VaR Magnitude", orientation="v",
        measure=["absolute"] + ["relative"] * len(scenarios_dict),
        x=names, y=values,
        connector=dict(line=dict(color="gray", dash="dot")),
        increasing=dict(marker_color="darkred"),
        decreasing=dict(marker_color="steelblue"),
        totals=dict(marker_color="maroon")
    ))
    fig.update_layout(
        title="Stress Scenario Waterfall: VaR Magnitude Escalation",
        yaxis_title="|VaR| (Daily Loss)"
    )
    return fig


try:
    prices, returns = load_data(ticker, *date_range)
    garch_result = fit_garch(returns) if use_garch else None

    # Grid search for Model Deep-Dive tab (single asset)
    from src.garch import fit_garch_grid
    garch_grid_result = fit_garch_grid(returns) if use_garch else None

    # VaR/ES for selected alpha (primary)
    result = compute_var_es(
        returns, method=method, alpha=alpha, horizon=horizon,
        garch_result=garch_result
    )

    # VaR/ES at all three confidence levels (for multi-alpha charts)
    alpha_results = {}
    for a in ALPHAS:
        g = garch_result if method != "historical" else None
        alpha_results[a] = compute_var_es(
            returns, method=method, alpha=a, garch_result=g
        )

    data_ok = True
except Exception as exc:
    data_ok = False
    error_msg = str(exc)
    st.error(f"**Data Error:** {error_msg}")
    st.info(
        "Try a different ticker or a narrower date range. "
        "Swedish equities (Ericsson, Volvo, H&M, Swedbank) use "
        "Yahoo Finance Stockholm suffix (.ST)."
    )

if not data_ok:
    st.stop()

# ── Tabs ─────────────────────────────────────────────────────────────────────

( tab_exec, tab_snapshot, tab_compare, tab_model,
  tab_methods, tab_backtest, tab_stress ) = st.tabs([
    "Executive Summary", "Risk Snapshot", "Method Comparison",
    "Model Deep-Dive", "Methodology", "Backtesting", "Stress Tests"
])

# ── Tab 1: Executive Summary ────────────────────────────────────────────────

with tab_exec:
    st.header("VaR & Expected Shortfall Risk Engine")

    st.markdown("""
    A **production-grade risk measurement and validation pipeline** for
    Swedish equities, built to FRTB (2019) and Basel III regulatory standards.

    **What this dashboard demonstrates:**
    - **Volatility modeling** — GARCH/EGARCH with grid search across
      specifications and distributions, selected by AICc
    - **Risk measurement** — VaR and Expected Shortfall via Historical,
      Parametric (Normal & Student-t), and Monte Carlo methods
    - **Model validation** — Kupiec, Christoffersen, and Acerbi-Szekely
      backtests with Basel Traffic Light classification
    - **Stress testing** — Historical scenario analysis and worst-window
      detection
    """)

    st.divider()

    # Spotlight metrics
    col1, col2, col3 = st.columns(3)
    col1.metric(
        "Best GARCH Model",
        f"{garch_result.vol}({garch_result.p},{garch_result.q})-{garch_result.dist}" if use_garch and garch_result else "N/A",
        delta=f"AICc: {garch_result.aicc:.1f}" if use_garch and garch_result else None,
        help="AICc-optimal volatility specification selected via grid search across 16 model combinations"
    )
    col2.metric(
        "Current VaR (97.5%)", f"{alpha_results[0.975].var:.4%}",
        delta=f"Method: {method.capitalize()}",
        help="1-day VaR at FRTB 97.5% confidence level"
    )
    col3.metric(
        "Data Coverage",
        f"{len(returns):,d} obs",
        delta=f"{date_range[0]} → {date_range[1]}",
        help="Daily log returns from Yahoo Finance"
    )

    st.divider()

    # Pipeline flow diagram
    st.subheader("Analysis Pipeline")

    pipe_cols = st.columns(5)
    pipe_steps = [
        ("📊", "Data\nExploration", "Stylised facts,\nfat tails, ADF"),
        ("📈", "GARCH\nModeling", "Grid search,\nEGARCH selection"),
        ("📉", "VaR & ES\nComputation", "4 methods,\n3 confidence levels"),
        ("✅", "Backtesting", "Kupiec, Christoffersen,\nAcerbi-Szekely"),
        ("⚠️", "Stress\nTesting", "Historical scenarios,\nworst window"),
    ]
    for col, (emoji, title, desc) in zip(pipe_cols, pipe_steps):
        with col:
            st.markdown(f"### {emoji}")
            st.markdown(f"**{title}**")
            st.caption(desc)

    st.info(
        "**Data provenance:** All analysis uses 5 Swedish equities "
        "(OMXS30 index + Ericsson, Volvo, H&M, Swedbank) with daily "
        "log returns from 2010–2025. The full analytical pipeline is "
        "documented in Jupyter notebooks (01–05) in the `notebooks/` "
        "directory."
    )

# ── Tab 2: Risk Snapshot ────────────────────────────────────────────────────

with tab_snapshot:
    st.header("Risk Snapshot")

    col1, col2, col3 = st.columns(3)
    col1.metric("VaR (1-Day)", f"{result.var:.4%}")
    col2.metric("ES (1-Day)", f"{result.es:.4%}")
    col3.metric(
        "Annualized Vol", f"{np.std(returns) * np.sqrt(252):.2%}"
    )

    fig = go.Figure()
    fig.add_trace(
        go.Histogram(
            x=returns, nbinsx=60, name="Returns", marker_color="lightblue"
        )
    )
    fig.add_vline(
        x=result.var,
        line_dash="dash",
        line_color="red",
        annotation_text=f"VaR {alpha:.1%}",
    )
    fig.add_vline(
        x=result.es,
        line_dash="dot",
        line_color="darkred",
        annotation_text=f"ES {alpha:.1%}",
    )
    fig.update_layout(
        title="Return Distribution with VaR/ES Overlay",
        xaxis_title="Log Return",
        bargap=0.05,
    )
    st.plotly_chart(fig, use_container_width=True)

    st.caption(
        f"Method: {result.method} | GARCH: {result.garch_used} | "
        f"α: {result.alpha} | Horizon: {result.horizon}d"
    )

    # GARCH model summary card
    if use_garch and garch_result and hasattr(garch_result, 'params'):
        with st.expander("GARCH Model Summary", expanded=False):
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Model", f"{garch_result.vol}({garch_result.p},{garch_result.q})")
            c2.metric("Distribution", garch_result.dist.capitalize())
            c3.metric("AICc", f"{garch_result.aicc:.1f}")
            # Persistence = alpha1 + beta1 (sum of ARCH + GARCH terms)
            alpha_keys = [k for k in garch_result.params if 'alpha' in k]
            beta_keys = [k for k in garch_result.params if 'beta' in k]
            persistence = sum(abs(garch_result.params[k]) for k in alpha_keys + beta_keys)
            half_life = np.log(0.5) / np.log(max(persistence, 0.001)) if 0 < persistence < 1 else float('inf')
            c4.metric("Half-Life", f"{half_life:.0f} days" if half_life < 1e6 else "∞")
            st.caption(
                f"**Interpretation:** Volatility shocks decay by 50% in "
                f"approximately {half_life:.0f} trading days. "
                f"The {garch_result.vol} specification captures "
                f"{'leverage effects' if garch_result.vol == 'EGARCH' else 'symmetric volatility responses'} — "
                f"negative returns {'increase' if garch_result.vol == 'EGARCH' else 'have the same impact on'} "
                f"future volatility as positive returns."
            )

    with st.expander("How to Read This Chart", expanded=False):
        st.markdown("""
        **Histogram:** Shows the empirical distribution of daily log returns.
        The shape reveals key risk features:
        - **Fat tails** (more observations at extremes than Normal predicts)
          → why parametric-Normal VaR underestimates risk
        - **Peaked center** → returns cluster near zero most days

        **VaR (red dashed line):** The loss threshold exceeded only
        $(1 - \\alpha) \\times 100\\%$ of the time. At 97.5%, expect
        2-3 breaches per 100 trading days.

        **ES (red dotted line):** The *average* loss on days when VaR is
        breached — always more severe than VaR. ES is the FRTB-preferred
        risk measure because it accounts for tail severity beyond the
        threshold.
        """)

    st.caption(
        "ES ≥ VaR (always): Expected Shortfall is mathematically "
        "guaranteed to be at least as large as VaR in absolute value "
        "— it measures the average of all tail losses, not just the "
        "threshold. This 'coherence' property makes ES the FRTB-standard "
        "risk measure."
    )

# ── Tab 3: Method Comparison ─────────────────────────────────────────────────

with tab_compare:
    st.header("Method Comparison")

    methods = ["historical", "parametric", "mc"]
    results = {}
    for m in methods:
        try:
            g = garch_result if m != "historical" else None
            results[m] = compute_var_es(
                returns, method=m, alpha=alpha, garch_result=g
            )
        except Exception:
            results[m] = None

    comp_data = []
    for m, r in results.items():
        if r is not None:
            comp_data.append(
                {
                    "Method": m.capitalize(),
                    "VaR": f"{r.var:.4%}",
                    "ES": f"{r.es:.4%}",
                    "VaR (abs)": abs(r.var),
                    "ES (abs)": abs(r.es),
                }
            )
    df_comp = pd.DataFrame(comp_data)
    st.dataframe(df_comp[["Method", "VaR", "ES"]], use_container_width=True)

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            name="VaR",
            x=df_comp["Method"],
            y=df_comp["VaR (abs)"],
            marker_color="steelblue",
        )
    )
    fig.add_trace(
        go.Bar(
            name="ES",
            x=df_comp["Method"],
            y=df_comp["ES (abs)"],
            marker_color="darkred",
        )
    )
    fig.update_layout(
        title="VaR vs ES by Method",
        yaxis_title="Loss Magnitude",
        barmode="group",
    )
    st.plotly_chart(fig, use_container_width=True)

# ── Tab 4: Model Deep-Dive ──────────────────────────────────────────────────

with tab_model:
    st.header("Model Deep-Dive")
    st.info(
        "GARCH residual diagnostics, QQ plots, and conditional "
        "volatility decomposition — coming in the next enhancement."
    )

# ── Tab 5: Methodology ──────────────────────────────────────────────────────

with tab_methods:
    st.header("Methodology Reference")
    st.info(
        "Detailed descriptions of VaR/ES calculation methods, "
        "distributional assumptions, and FRTB framework — coming "
        "in the next enhancement."
    )

# ── Tab 6: Backtesting ──────────────────────────────────────────────────────

with tab_backtest:
    st.header("Backtesting")

    est_window = 500
    if len(returns) < est_window:
        st.warning(
            f"Need at least {est_window} observations for backtesting "
            f"(have {len(returns)}). Select a wider date range."
        )
        st.stop()

    var_fc, es_fc, real = [], [], []
    var_fc_975 = []

    for t in range(est_window, len(returns)):
        train = returns[t - est_window : t]
        g = fit_garch(train) if use_garch else None
        r = compute_var_es(train, method=method, alpha=alpha, garch_result=g)
        var_fc.append(r.var)
        es_fc.append(r.es)
        real.append(returns[t])
        # FRTB: also track 97.5% VaR breaches
        r_975 = compute_var_es(train, method=method, alpha=0.975, garch_result=g)
        var_fc_975.append(r_975.var)

    var_arr = np.array(var_fc)
    es_arr = np.array(es_fc)
    ret_arr = np.array(real)
    var_975_arr = np.array(var_fc_975)
    breaches = (ret_arr <= var_arr).astype(int)
    breaches_975 = (ret_arr <= var_975_arr).astype(int)

    k_test = kupiec_test(breaches.sum(), len(breaches), alpha)
    c_test = christoffersen_test(breaches)
    z2_test = acerbi_szekely_z2(ret_arr, var_arr, es_arr, alpha, n_sim=500)
    tl_basel = traffic_light(breaches.sum(), len(breaches), framework="basel1996")
    tl_frtb = traffic_light(
        breaches_99=breaches.sum(),
        breaches_975=breaches_975.sum(),
        total=len(breaches),
        framework="frtb2019",
    )

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Breaches", f"{breaches.sum()}/{len(breaches)}")
    col2.metric("Kupiec p-value", f"{k_test.p_value:.4f}")
    col3.metric("Christoffersen p-value", f"{c_test.p_value:.4f}")
    col4.metric("Acerbi-Szekely Z2", f"{z2_test.p_value:.4f}")

    st.subheader("Traffic Light")
    c1, c2 = st.columns(2)
    color = {"green": "🟢", "yellow": "🟡", "red": "🔴"}
    c1.metric(
        "Basel 1996",
        f"{color[tl_basel['zone']]} {tl_basel['zone']}",
        delta=f"k={tl_basel['multiplier']}",
    )
    c2.metric(
        "FRTB 2019",
        f"{color[tl_frtb['zone']]} {tl_frtb['zone']}",
        delta=f"99%: {breaches.sum()}  |  97.5%: {breaches_975.sum()}",
    )

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            y=ret_arr, name="Returns", line=dict(color="gray", width=1)
        )
    )
    fig.add_trace(
        go.Scatter(
            y=var_arr,
            name="VaR Forecast",
            line=dict(color="red", dash="dash"),
        )
    )
    breach_idx = np.where(breaches)[0]
    fig.add_trace(
        go.Scatter(
            x=breach_idx,
            y=ret_arr[breach_idx],
            mode="markers",
            name="Breaches",
            marker=dict(color="red", size=8, symbol="x"),
        )
    )
    fig.update_layout(
        title="Rolling Backtest", xaxis_title="Day", yaxis_title="Return"
    )
    st.plotly_chart(fig, use_container_width=True)

# ── Tab 7: Stress Tests ──────────────────────────────────────────────────────

with tab_stress:
    st.header("Stress Tests")

    ret_series = pd.Series(
        returns,
        index=pd.date_range(
            start=date_range[0], periods=len(returns), freq="B"
        ),
    )

    scenarios = {}
    for label, s, e in [
        ("COVID 2020", "2020-02-19", "2020-03-23"),
        ("2008 GFC", "2008-09-01", "2008-12-31"),
    ]:
        try:
            scenarios[label] = run_historical_scenario(
                ret_series, s, e, label
            )
        except (ValueError, KeyError):
            pass

    try:
        worst_start, worst_end = find_worst_window(ret_series)
        scenarios["Worst 12-Month"] = run_historical_scenario(
            ret_series,
            str(worst_start.date()),
            str(worst_end.date()),
            "Worst Auto-Detected",
        )
    except (ValueError, KeyError):
        pass

    if scenarios:
        df_scenarios = pd.DataFrame(
            [
                {
                    "Scenario": s,
                    "VaR": f"{r.var:.4%}",
                    "ES": f"{r.es:.4%}",
                    "Cumulative P&L": f"{r.pnl:.4%}",
                    "Worst Day": f"{r.worst_day:.4%}",
                }
                for s, r in scenarios.items()
            ]
        )
        st.dataframe(df_scenarios, use_container_width=True)

        fig = go.Figure()
        for s, r in scenarios.items():
            fig.add_trace(
                go.Bar(name=s, x=["VaR", "ES"], y=[abs(r.var), abs(r.es)])
            )
        fig.update_layout(
            title="Stress Scenario Comparison",
            yaxis_title="Loss Magnitude",
            barmode="group",
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning(
            "No stress scenarios available for selected date range."
        )
