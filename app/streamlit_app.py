"""VaR & Expected Shortfall Risk Engine — Streamlit Dashboard."""

import os
import pickle
import sys
from pathlib import Path

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.backtest import (acerbi_szekely_z2, christoffersen_test, kupiec_test,
                          traffic_light)
from src.garch import fit_garch
from src.stress_test import find_worst_window, run_historical_scenario
from src.utils import compute_returns, fetch_prices

from src.var_methods import compute_var_es

from app.charts import (plot_breach_timeline, plot_conditional_vol,
                        plot_distribution_overlay, plot_news_impact_curve,
                        plot_qq_residuals, plot_scenario_waterfall)

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

# ── Session state init (must precede any st.session_state access) ────────

if "data_source" not in st.session_state:
    st.session_state.data_source = "default"

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

    st.divider()

    if st.button("Run Live Analysis", type="primary", use_container_width=True):
        st.session_state.data_source = "live"
        st.rerun()

    if st.session_state.data_source == "default":
        st.info(
            "**Default View:** Pre-computed OMXS30 results (2010–2025). "
            "Adjust controls above and click **Run Live Analysis**."
        )


@st.cache_data(ttl=3600)
def load_data(ticker, start, end):
    prices = fetch_prices(ticker, str(start), str(end))
    returns = compute_returns(prices, method="log")
    return prices, returns


# ── Helper Functions ────────────────────────────────────────────────────────


def _volatility_persistence(params):
    """Compute GARCH volatility persistence and half-life from parameters.

    Persistence = sum of all ARCH (alpha) and GARCH (beta) coefficients.
    Half-life = ln(0.5) / ln(persistence) trading days for mean-reversion.
    """
    alpha_keys = [k for k in params if "alpha" in k]
    beta_keys = [k for k in params if "beta" in k]
    persistence = sum(abs(params[k]) for k in alpha_keys + beta_keys)
    half_life = (
        np.log(0.5) / np.log(max(persistence, 0.001))
        if 0 < persistence < 1
        else float("inf")
    )
    return persistence, half_life

@st.cache_data(ttl=3600)
def _compute_defaults():
    """Pre-compute OMXS30 2010-2025 results for fallback display.

    Runs GARCH grid search, rolling backtest (500-day window, historical
    method at 99% VaR), and stress scenarios once per hour. Returns a
    dict with keys: garch_grid, garch_single, backtest, stress_scenarios,
    returns.
    """
    default_ticker = "^OMX"
    default_start = "2010-01-01"
    default_end = "2025-12-31"

    prices = fetch_prices(default_ticker, default_start, default_end)
    returns = compute_returns(prices, method="log")

    # GARCH: grid search for best model + single fit for cond vol
    garch_grid = fit_garch_grid(returns)
    garch_single = fit_garch(returns)

    # Rolling backtest: 500-day window, historical method, 99% VaR
    est_window = 500
    var_fc, es_fc, real = [], [], []
    var_fc_975 = []
    prev_garch = None
    for t in range(est_window, len(returns)):
        train = returns[t - est_window : t]
        try:
            g = fit_garch(train)
            prev_garch = g
        except Exception:
            g = prev_garch  # reuse previous successful fit
        try:
            r = compute_var_es(
                train, method="historical", alpha=0.99, garch_result=g
            )
            var_fc.append(r.var)
            es_fc.append(r.es)
            real.append(returns[t])
            r_975 = compute_var_es(
                train, method="historical", alpha=0.975, garch_result=g
            )
            var_fc_975.append(r_975.var)
        except Exception:
            continue

    var_arr = np.array(var_fc)
    es_arr = np.array(es_fc)
    ret_arr = np.array(real)
    var_975_arr = np.array(var_fc_975)
    breaches = (ret_arr <= var_arr).astype(int)
    breaches_975 = (ret_arr <= var_975_arr).astype(int)

    # Stress scenarios
    ret_series = pd.Series(
        returns,
        index=pd.date_range(
            start=pd.Timestamp(default_start),
            periods=len(returns),
            freq="B",
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

    backtest = {
        "breaches_sum": int(breaches.sum()),
        "breaches_total": len(breaches),
        "breaches_975_sum": int(breaches_975.sum()),
        "kupiec": kupiec_test(breaches.sum(), len(breaches), 0.99),
        "christoffersen": christoffersen_test(breaches),
        "acerbi_szekely": acerbi_szekely_z2(
            ret_arr, var_arr, es_arr, 0.99, n_sim=500
        ),
        "traffic_basel": traffic_light(
            breaches.sum(), len(breaches), framework="basel1996"
        ),
        "traffic_frtb": traffic_light(
            breaches_99=breaches.sum(),
            breaches_975=breaches_975.sum(),
            total=len(breaches),
            framework="frtb2019",
        ),
        "ret_arr": ret_arr,
        "var_arr": var_arr,
        "es_arr": es_arr,
        "breaches_arr": breaches,
        "breaches_975_arr": breaches_975,
    }

    # VaR/ES at all three confidence levels (historical, GARCH-scaled)
    alpha_results_default = {}
    for a in ALPHAS:
        alpha_results_default[a] = compute_var_es(
            returns, method="historical", alpha=a, horizon=1,
            garch_result=garch_single
        )

    return {
        "garch_grid": garch_grid,
        "garch_single": garch_single,
        "backtest": backtest,
        "stress_scenarios": scenarios,
        "returns": returns,
        "alpha_results": alpha_results_default,
        "result": alpha_results_default[0.975],
    }


# ── Load pre-computed defaults from pickle ────────────────────────────────

_DEFAULTS_PATH = Path(__file__).parent / "defaults.pkl"

def _load_defaults():
    with open(_DEFAULTS_PATH, "rb") as f:
        return pickle.load(f)

defaults = _load_defaults()


# ── Live data loading (only when user triggers analysis) ──────────────────

if st.session_state.data_source == "live":
    try:
        prices, returns = load_data(ticker, *date_range)
        garch_result = fit_garch(returns) if use_garch else None

        from src.garch import fit_garch_grid
        garch_grid_result = fit_garch_grid(returns) if use_garch else None

        result = compute_var_es(
            returns, method=method, alpha=alpha, horizon=horizon,
            garch_result=garch_result
        )

        alpha_results = {}
        for a in ALPHAS:
            g = garch_result if method != "historical" else None
            alpha_results[a] = compute_var_es(
                returns, method=method, alpha=a, horizon=horizon,
                garch_result=g
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


# ── Set active data source for tabs ───────────────────────────────────────

using_default = st.session_state.data_source == "default"

if using_default:
    returns = defaults["returns"]
    result = defaults["result"]
    alpha_results = defaults["alpha_results"]
    garch_result = defaults["garch_single"]
    garch_grid_result = defaults["garch_grid"]

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
    if use_garch and garch_result:
        col1.metric(
            "GARCH Model",
            f"{garch_result.vol}({garch_result.p},{garch_result.q})-{garch_result.dist}",
            delta=f"AICc: {garch_result.aicc:.1f}",
            help="GARCH(1,1) volatility model. See Model Deep-Dive tab for grid search across 16 specifications."
        )
    else:
        col1.metric(
            "GARCH Model",
            "N/A",
            help="Enable 'Use GARCH volatility' in sidebar to fit a model."
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
            persistence, half_life = _volatility_persistence(garch_result.params)
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

    st.divider()

    # Distribution overlay for current selection
    st.subheader("Distribution Fit Analysis")
    dist_fig = plot_distribution_overlay(returns, result, alpha)
    st.plotly_chart(dist_fig, use_container_width=True)

    with st.expander("Why Distribution Choice Matters"):
        st.markdown(f"""
        **Normal distribution underestimates tail risk.** The empirical
        returns have excess kurtosis (fat tails) — extreme events occur
        more frequently than the Normal distribution predicts.

        At {alpha:.1%} confidence:
        - **Parametric-Normal VaR:** {alpha_results[alpha].var:.4%}
        - **Parametric-t VaR:** use Method Comparison table above for t-based estimate

        The Student-t distribution better captures tail fatness through
        its degrees-of-freedom parameter (ν). Lower ν = fatter tails.
        Typical equity returns have ν between 3 and 6.
        """)

    # Confidence level sensitivity
    st.subheader("Confidence Level Sensitivity")
    sens_data = {
        "Confidence": [f"{a:.1%}" for a in ALPHAS],
        "VaR": [abs(alpha_results[a].var) for a in ALPHAS],
        "ES": [abs(alpha_results[a].es) for a in ALPHAS],
    }
    fig_sens = go.Figure()
    fig_sens.add_trace(go.Bar(
        name="VaR", x=sens_data["Confidence"], y=sens_data["VaR"],
        marker_color="steelblue"
    ))
    fig_sens.add_trace(go.Bar(
        name="ES", x=sens_data["Confidence"], y=sens_data["ES"],
        marker_color="darkred"
    ))
    fig_sens.update_layout(
        title="VaR vs ES Across Confidence Levels",
        yaxis_title="Loss Magnitude", barmode="group"
    )
    st.plotly_chart(fig_sens, use_container_width=True)

    st.caption(
        "As confidence level increases (95% → 97.5% → 99%), both VaR "
        "and ES become more conservative. ES grows faster than VaR at "
        "higher confidence levels because it captures more of the tail."
    )

# ── Tab 4: Model Deep-Dive ──────────────────────────────────────────────────

with tab_model:
    st.header("GARCH Model Deep-Dive")

    if not use_garch:
        st.info("Enable 'Use GARCH volatility' in the sidebar to see model diagnostics.")
    else:
        # Determine data source: live or default
        gr = garch_grid_result
        if gr is None:
            gr = defaults["garch_grid"]

        st.subheader("Model Selection: Grid Search Results")

        with st.expander("How Model Selection Works", expanded=False):
            st.markdown("""
            **Grid search** evaluates every combination of:
            - **Volatility model:** GARCH (symmetric) vs EGARCH (leverage)
            - **Error distribution:** Normal vs Student-t
            - **Lag orders:** p (ARCH) ∈ {1,2}, q (GARCH) ∈ {1,2}

            **AICc (Corrected Akaike Information Criterion)** penalizes
            model complexity — lower AICc = better fit without overfitting.
            AICc is preferred over AIC for financial data because it
            includes a finite-sample correction.
            """)

        st.markdown(f"""
        **Selected Model:** **{gr.vol}({gr.p},{gr.q})-{gr.dist}**
        | AICc: {gr.aicc:.1f} | AIC: {gr.aic:.1f} | BIC: {gr.bic:.1f}
        """)

        if gr.params:
            param_df = pd.DataFrame(
                {"Value": list(gr.params.values())},
                index=list(gr.params.keys())
            )
            st.dataframe(param_df, use_container_width=True)

        persistence, half_life = _volatility_persistence(gr.params)

        col1, col2 = st.columns(2)
        col1.metric(
            "Volatility Persistence", f"{persistence:.4f}",
            delta="Stationary" if persistence < 1 else "Non-stationary"
        )
        col2.metric(
            "Half-Life",
            f"{half_life:.0f} trading days" if half_life < 1e6 else "∞",
            delta=f"~{half_life/252:.1f} years" if half_life < 1e6 else None
        )

        # Conditional volatility plot
        st.subheader("Conditional Volatility")
        st.markdown(
            "**Context:** The GARCH model decomposes returns into a "
            "time-varying volatility component. Spikes correspond to "
            "market stress events."
        )
        try:
            display_name = NAME_MAP[ticker] if not using_default else "OMXS30 (default)"
            display_date_range = date_range if not using_default else (
                pd.Timestamp("2010-01-01"), pd.Timestamp("2025-12-31"))
            vol_fig = plot_conditional_vol(gr, display_name, display_date_range)
            st.plotly_chart(vol_fig, use_container_width=True)
        except Exception as e:
            st.warning(f"Could not render conditional volatility plot: {e}")

        # Standardized residuals
        st.subheader("Model Diagnostics: Standardized Residuals")
        if hasattr(gr, 'cond_vol') and len(gr.cond_vol) > 1:
            # Get returns matching the cond_vol length for defaults
            resid_returns = (
                returns[-len(gr.cond_vol):] if not using_default
                else defaults["returns"][-len(gr.cond_vol):]
            )
            std_resid = resid_returns / np.maximum(gr.cond_vol, 1e-10)

            col1, col2 = st.columns(2)
            with col1:
                resid_display_name = display_name if not using_default else "OMXS30"
                qq_fig = plot_qq_residuals(std_resid, gr.dist, resid_display_name)
                st.plotly_chart(qq_fig, use_container_width=True)

            with col2:
                from statsmodels.tsa.stattools import acf
                sq_resid = std_resid ** 2
                acf_vals = acf(sq_resid, nlags=20)
                acf_fig = go.Figure()
                acf_fig.add_trace(go.Bar(
                    x=list(range(1, 21)), y=acf_vals[1:],
                    marker_color="steelblue"
                ))
                ci = 1.96 / np.sqrt(len(sq_resid))
                acf_fig.add_hline(
                    y=ci, line_dash="dash", line_color="red",
                    annotation_text="95% CI"
                )
                acf_fig.add_hline(
                    y=-ci, line_dash="dash", line_color="red"
                )
                acf_fig.update_layout(
                    title="ACF of Squared Std Residuals",
                    xaxis_title="Lag", yaxis_title="Autocorrelation"
                )
                st.plotly_chart(acf_fig, use_container_width=True)

            st.caption(
                "**Interpretation:** If the GARCH model adequately "
                "captures volatility dynamics, standardized residuals "
                "should be approximately i.i.d. — QQ points should lie "
                "near the diagonal, and squared residuals should show "
                "no significant autocorrelation (bars within the 95% CI)."
            )

        # News impact curve (new)
        st.subheader("News Impact Curve")
        with st.expander("What is the News Impact Curve?"):
            st.markdown("""
            The **news impact curve** (Engle & Ng, 1993) shows how
            yesterday's return shock (ε_{t-1}) affects today's
            volatility forecast (σ²_t).

            **For GARCH:** The curve is symmetric — positive and negative
            shocks of equal size produce identical volatility increases.
            This assumes no leverage effect.

            **For EGARCH:** The curve is asymmetric — negative shocks
            (market declines) produce larger volatility increases than
            positive shocks (market gains) of the same magnitude.
            This is the **leverage effect**: bad news increases future
            volatility more than good news.

            **γ < 0** → negative shocks amplify volatility more.
            This is the canonical pattern for equity markets.
            """)
        try:
            nic_fig = plot_news_impact_curve(gr)
            st.plotly_chart(nic_fig, use_container_width=True)
        except Exception as e:
            st.warning(f"Could not render news impact curve: {e}")

        # EGARCH leverage explainer
        if gr.vol == "EGARCH":
            st.subheader("Leverage Effect (EGARCH)")
            gamma_keys = [k for k in gr.params if 'gamma' in k]
            if gamma_keys:
                gamma_val = gr.params[gamma_keys[0]]
                st.markdown(f"""
                **Asymmetric volatility parameter (γ):** {gamma_val:.4f}

                **Interpretation:**
                - γ < 0 → negative returns increase volatility more than
                  positive returns (the **leverage effect**)
                - |γ| measures the magnitude of the asymmetry
                - This parameter distinguishes EGARCH from standard GARCH
                  — it explicitly models the empirical observation that
                  volatility rises more after market declines than after
                  equivalent gains

                **Finding (from notebooks):** EGARCH was selected for all
                five Swedish equities by AICc. The leverage effect is a
                systematic feature of this equity universe — negative
                shocks amplify future volatility 2-3x more than positive
                shocks of equal magnitude.
                """)

        st.info(
            "**Notebook reference:** Full grid search details and "
            "per-asset diagnostics in Notebook 02 — GARCH Volatility "
            "Modeling (`notebooks/02_garch_volatility.ipynb`)."
        )

# ── Tab 5: Methodology ──────────────────────────────────────────────────────

with tab_methods:
    st.header("VaR & ES Methodology")

    st.markdown("""
    Three complementary approaches for estimating Value-at-Risk and
    Expected Shortfall, each with different assumptions and trade-offs.
    """)

    method_tabs = st.tabs(["Historical Simulation", "Parametric", "Monte Carlo"])

    with method_tabs[0]:
        st.subheader("Historical Simulation")
        st.markdown("""
        **How it works:**
        1. Take historical returns as the empirical distribution
        2. Sort returns from worst to best
        3. VaR = the $(1-\\alpha)$-th percentile of sorted returns
        4. ES = average of all returns below the VaR threshold

        **Pros:** No distribution assumptions. Non-parametric. Captures
        actual tail shape including fat tails and skewness.

        **Cons:** Cannot extrapolate beyond historical extremes.
        Sensitive to the lookback window. Assumes the past distribution
        is representative of the future.

        **Best for:** Regulatory reporting where conservatism and
        transparency are valued over statistical efficiency.
        """)

    with method_tabs[1]:
        st.subheader("Parametric (Variance-Covariance)")
        st.markdown("""
        **How it works:**
        1. Assume returns follow a specific distribution (Normal or
           Student-t)
        2. Estimate parameters (μ, σ) from the data
        3. VaR = μ + z_α × σ   (where z_α is the quantile)
        4. ES = μ − [φ(z_α)/(1−α)] × σ

        **Normal assumption:**
        - Computationally trivial — closed-form solution
        - **Underestimates tail risk** when returns are fat-tailed
        - Works well for short horizons with near-Normal returns

        **Student-t assumption:**
        - Accommodates fat tails via the degrees-of-freedom parameter
        - Requires numerical MLE for parameter estimation
        - **Preferred for equity returns** where excess kurtosis is
          present (notebooks confirm ν ∈ [3.6, 7.6] for Swedish equities)

        **Where the VaR-ES gap comes from:** Under Normal distribution,
        ES/VaR ≈ 1.15 at 99%. Under Student-t with ν=4, ES/VaR ≈ 1.35.
        The gap grows as tails get fatter — a critical regulatory
        consideration.
        """)

    with method_tabs[2]:
        st.subheader("Monte Carlo Simulation")
        st.markdown("""
        **How it works:**
        1. Simulate N random return paths from an assumed distribution
           (typically Geometric Brownian Motion)
        2. Sort simulated terminal returns
        3. VaR = percentile of simulated distribution
        4. ES = average of tail outcomes

        **Implementation detail:** This dashboard uses **antithetic
        variates** for variance reduction — each random draw is paired
        with its negative mirror, reducing Monte Carlo error by ~50%
        at no extra computational cost.

        **Pros:** Most flexible. Can incorporate complex dependencies,
        non-linear instruments, and multi-horizon paths.

        **Cons:** Computationally expensive. Results depend on the
        assumed stochastic process. Simulation noise means estimates
        vary between runs.
        """)

    st.divider()

    # ES coherence explainer
    st.subheader("Why ES Replaced VaR in FRTB")
    with st.expander("ES Coherence — The Mathematical Argument"):
        st.markdown("""
        A **coherent risk measure** (Artzner et al., 1999) must satisfy
        four axioms:

        1. **Monotonicity:** If portfolio A always outperforms B,
           A has lower risk
        2. **Sub-additivity:** Risk(A + B) ≤ Risk(A) + Risk(B) —
           diversification should not increase risk
        3. **Positive homogeneity:** Doubling position size doubles risk
        4. **Translation invariance:** Adding cash reduces risk by
           that amount

        **VaR violates sub-additivity.** Two positions can have a
        combined VaR that exceeds the sum of their individual VaRs —
        penalizing diversification. This is especially problematic for
        portfolios with non-linear instruments or concentrated tail risk.

        **ES satisfies all four axioms.** It is always sub-additive,
        making it the **Basel III / FRTB standard** for market risk
        capital calculation.

        **Empirical evidence from notebooks:** The equally-weighted
        portfolio of 5 Swedish equities shows VaR sub-additivity
        violations of 4.6–7.8%, while ES shows no violation —
        confirming why regulators moved to ES.
        """)

    st.divider()

    # Method selection guide
    st.subheader("Method Selection Guide")
    guide_data = pd.DataFrame({
        "Criterion": ["Accuracy (fat tails)", "Computation speed",
                       "Regulatory acceptance", "Forward-looking",
                       "Requires distribution assumption"],
        "Historical": ["★★★★★", "★★★★★", "★★★★☆", "✗", "✗"],
        "Parametric-t": ["★★★★☆", "★★★★★", "★★★★☆", "✗", "✓"],
        "Parametric-Normal": ["★★☆☆☆", "★★★★★", "★★☆☆☆", "✗", "✓"],
        "Monte Carlo": ["★★★☆☆", "★★☆☆☆", "★★★☆☆", "✓", "✓"],
    })
    st.dataframe(guide_data, use_container_width=True, hide_index=True)

# ── Tab 6: Backtesting ──────────────────────────────────────────────────────

with tab_backtest:
    st.header("Backtesting")

    est_window = 500
    loop_failed = False
    local_fallback = using_default

    # Decide data source
    if using_default:
        # Skip live loop entirely \u2014 use pre-computed defaults
        pass
    elif len(returns) < est_window:
        local_fallback = True
    else:
        # Attempt live rolling backtest
        var_fc, es_fc, real = [], [], []
        var_fc_975 = []
        prev_garch = None
        for t in range(est_window, len(returns)):
            train = returns[t - est_window : t]
            try:
                g = fit_garch(train) if use_garch else None
                prev_garch = g
            except Exception:
                g = prev_garch
                loop_failed = True
            try:
                r = compute_var_es(
                    train, method=method, alpha=alpha, garch_result=g
                )
                var_fc.append(r.var)
                es_fc.append(r.es)
                real.append(returns[t])
                r_975 = compute_var_es(
                    train, method=method, alpha=0.975, garch_result=g
                )
                var_fc_975.append(r_975.var)
            except Exception:
                loop_failed = True
                continue

        if len(var_fc) == 0:
            local_fallback = True

    if local_fallback:
        bt = defaults["backtest"]
        var_arr = bt["var_arr"]
        es_arr = bt["es_arr"]
        ret_arr = bt["ret_arr"]
        var_975_arr = bt["breaches_975_arr"]
        breaches = bt["breaches_arr"]
        breaches_975 = bt["breaches_975_arr"]
        k_test = bt["kupiec"]
        c_test = bt["christoffersen"]
        z2_test = bt["acerbi_szekely"]
        tl_basel = bt["traffic_basel"]
        tl_frtb = bt["traffic_frtb"]
    else:
        var_arr = np.array(var_fc)
        es_arr = np.array(es_fc)
        ret_arr = np.array(real)
        var_975_arr = np.array(var_fc_975)
        breaches = (ret_arr <= var_arr).astype(int)
        breaches_975 = (ret_arr <= var_975_arr).astype(int)

        k_test = kupiec_test(breaches.sum(), len(breaches), alpha)
        c_test = christoffersen_test(breaches)
        z2_test = acerbi_szekely_z2(
            ret_arr, var_arr, es_arr, alpha, n_sim=500
        )
        tl_n = min(len(breaches), 250)
        tl_basel = traffic_light(
            int(breaches[-tl_n:].sum()), tl_n, framework="basel1996"
        )
        tl_frtb = traffic_light(
            breaches_99=int(breaches[-tl_n:].sum()),
            breaches_975=int(breaches_975[-tl_n:].sum()),
            total=tl_n,
            framework="frtb2019",
        )

        if loop_failed:
            st.warning(
                "Some GARCH fits in the rolling loop failed to converge. "
                "Results may use fallback estimates for those windows."
            )

    # Metrics row
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Breaches", f"{breaches.sum()}/{len(breaches)}")
    col2.metric("Kupiec p-value", f"{k_test.p_value:.4f}")
    col3.metric("Christoffersen p-value", f"{c_test.p_value:.4f}")
    col4.metric("Acerbi-Szekely Z2", f"{z2_test.p_value:.4f}")

    # Traffic light
    st.subheader("Traffic Light")
    c1, c2 = st.columns(2)
    color = {"green": "\U0001f7e2", "yellow": "\U0001f7e1", "red": "\U0001f534"}
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

    # Rolling backtest chart
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

    # Test interpretation
    st.subheader("Test Interpretation")
    with st.expander("What These Tests Actually Mean"):
        st.markdown(f"""
        **Kupiec POF Test** (p = {k_test.p_value:.4f}):
        Tests whether the number of VaR breaches matches expectation.
        - H\u2080: breach rate = expected rate ({1-alpha:.1%})
        - p > 0.05 \u2192 **pass** \u2014 breaches are at acceptable frequency
        - p < 0.05 \u2192 **fail** \u2014 model is miscalibrated
        - Result: **{'PASS \u2713' if not k_test.reject else 'FAIL \u2717'}**

        **Christoffersen Test** (p = {c_test.p_value:.4f}):
        Tests whether breaches are independent (not clustered).
        - H\u2080: breaches are independent over time
        - p > 0.05 \u2192 **pass** \u2014 breaches are randomly scattered
        - p < 0.05 \u2192 **fail** \u2014 breaches cluster, model slow to adapt
        - Result: **{'PASS \u2713' if not c_test.reject else 'FAIL \u2717'}**

        **Acerbi-Szekely Z2 Test** (p = {z2_test.p_value:.4f}):
        Tests whether ES forecasts are well-specified.
        - H\u2080: ES forecasts correctly capture tail severity
        - p > 0.05 \u2192 **pass** \u2014 ES estimates are adequate
        - p < 0.05 \u2192 **fail** \u2014 ES is miscalibrated
        - Result: **{'PASS \u2713' if not z2_test.reject else 'FAIL \u2717'}**
        """)

    # FRTB dual-condition breach map
    st.subheader("FRTB Dual-Condition Breach Map")
    breach_fig = plot_breach_timeline(ret_arr, breaches, breaches_975)
    st.plotly_chart(breach_fig, use_container_width=True)

    st.caption(
        "FRTB (2019) requires backtesting at both 99% and 97.5% "
        "confidence: green zone if \u226412 breaches at 99% AND \u226430 at "
        "97.5% over 250 days. Red crosses = 99% breaches. Orange "
        "triangles = 97.5% breaches only."
    )

    # Traffic light summary
    st.subheader("Regulatory Capital Multiplier")
    reg_col1, reg_col2 = st.columns(2)
    with reg_col1:
        st.metric(
            "Basel 1996",
            f"{color[tl_basel['zone']]} {tl_basel['zone'].upper()}",
            delta=f"Multiplier: {tl_basel['multiplier']}x"
        )
        st.caption(
            "Basel I market risk amendment. Green zone: k=3.0 (minimum). "
            "Yellow zone: k=3.4\u20133.85 (capital add-on 13\u201328%). "
            "Red zone: k=4.0 (maximum penalty)."
        )
    with reg_col2:
        st.metric(
            "FRTB 2019",
            f"{color[tl_frtb['zone']]} {tl_frtb['zone'].upper()}",
            delta=f"99%: {breaches.sum()} | 97.5%: {breaches_975.sum()}"
        )
        st.caption(
            "FRTB dual-condition test. Green zone requires BOTH: "
            "\u226412 breaches at 99% AND \u226430 breaches at 97.5% (over 250 days). "
            "More stringent than Basel 1996 \u2014 single condition failure "
            "triggers red zone."
        )

# ── Tab 7: Stress Tests ──────────────────────────────────────────────────────

with tab_stress:
    st.header("Stress Tests")

    local_fallback = using_default

    if using_default:
        scenarios = defaults["stress_scenarios"]
    else:
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

        if not scenarios:
            scenarios = defaults["stress_scenarios"]
            local_fallback = True

    if local_fallback:
        st.info(
            "**Showing default view (OMXS30, 2010–2025).** "
            "No stress scenarios overlap the selected date range. "
            "Select a range covering 2008–2020 for live analysis."
        )

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

    st.divider()

    # Waterfall chart
    st.subheader("Stress Escalation Waterfall")
    try:
        baseline_va = abs(alpha_results[0.975].var) if not local_fallback else abs(
            np.percentile(defaults["returns"], 2.5)
        )
        waterfall_fig = plot_scenario_waterfall(scenarios, baseline_va)
        st.plotly_chart(waterfall_fig, use_container_width=True)
    except Exception as e:
        st.warning(f"Could not render waterfall chart: {e}")

    st.caption(
        "The waterfall shows how VaR magnitude escalates from baseline "
        "(full-sample) → crisis scenarios → worst historical window. "
        "Each step represents the additional risk revealed by stress "
        "testing beyond normal-market VaR."
    )

    # Regulatory context
    st.subheader("Regulatory Context")
    with st.expander("How Stress Tests Feed Into Capital Requirements"):
        st.markdown("""
        **Basel III / FRTB Framework:**

        1. **VaR-based capital** (day-to-day): Uses the VaR model
           calibrated to a 1-year window with 97.5% ES

        2. **Stressed VaR / ES** (crisis-period calibration): Computed
           over a 12-month period of significant financial stress. In
           this dashboard: the COVID-19 crash (Feb–Mar 2020) or the
           worst auto-detected window.

        3. **Capital floor:** The higher of:
           - Current VaR × multiplier (3.0–4.0, from traffic light)
           - Stressed VaR × multiplier
           - Standardized Approach (SA-CCR) capital charge

        **Key insight:** Stress testing is not optional — it directly
        determines regulatory capital. A model that passes backtesting
        in normal markets but fails under stress produces capital
        requirements that are 1.5–3x higher.
        """)

    # Severity ranking
    st.subheader("Scenario Severity Ranking")
    severity_data = pd.DataFrame([
        {"Scenario": s, "|VaR|": abs(r.var), "|ES|": abs(r.es),
         "Cumul. P&L": r.pnl, "Worst Day": r.worst_day}
        for s, r in scenarios.items()
    ]).sort_values("|VaR|", ascending=False)
    st.dataframe(severity_data, use_container_width=True)

    st.caption(
        "**Finding (from notebooks):** Crisis-period VaR is 2–5x "
        "higher than full-sample VaR. COVID-19 crash produced the "
        "largest volatility spike in the 2010–2025 sample, with a "
        "5.3x VaR multiplier. The worst auto-detected window captures "
        "prolonged drawdowns that single-crash scenarios miss."
    )