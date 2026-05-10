"""VaR & Expected Shortfall Risk Engine — Streamlit Dashboard."""

import sys

sys.path.append(".")

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

st.set_page_config(page_title="VaR & ES Engine", layout="wide")
st.title("VaR & Expected Shortfall Risk Engine")
st.caption("FRTB-aligned | Basel III | Multi-asset risk analytics")

# ── Sidebar ──────────────────────────────────────────────────────────────────

with st.sidebar:
    st.header("Controls")
    ticker = st.selectbox(
        "Asset", ["^OMX", "^GSPC", "AAPL", "MSFT", "EURSEK=X"]
    )
    alpha = st.selectbox(
        "Confidence Level",
        [0.95, 0.975, 0.99],
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


prices, returns = load_data(ticker, *date_range)

garch_result = fit_garch(returns) if use_garch else None
result = compute_var_es(
    returns, method=method, alpha=alpha, horizon=horizon, garch_result=garch_result
)

# ── Tabs ─────────────────────────────────────────────────────────────────────

tab1, tab2, tab3, tab4 = st.tabs(
    ["Risk Snapshot", "Method Comparison", "Backtesting", "Stress Tests"]
)

# ── Tab 1: Risk Snapshot ─────────────────────────────────────────────────────

with tab1:
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

# ── Tab 2: Method Comparison ─────────────────────────────────────────────────

with tab2:
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

# ── Tab 3: Backtesting ───────────────────────────────────────────────────────

with tab3:
    st.header("Backtesting")

    est_window = 500
    var_fc, es_fc, real = [], [], []

    for t in range(est_window, len(returns)):
        train = returns[t - est_window : t]
        g = fit_garch(train) if use_garch else None
        r = compute_var_es(train, method=method, alpha=alpha, garch_result=g)
        var_fc.append(r.var)
        es_fc.append(r.es)
        real.append(returns[t])

    var_arr = np.array(var_fc)
    es_arr = np.array(es_fc)
    ret_arr = np.array(real)
    breaches = (ret_arr <= var_arr).astype(int)

    k_test = kupiec_test(breaches.sum(), len(breaches), alpha)
    c_test = christoffersen_test(breaches)
    z2_test = acerbi_szekely_z2(ret_arr, var_arr, es_arr, alpha, n_sim=500)
    tl_basel = traffic_light(breaches.sum(), len(breaches), framework="basel1996")
    tl_frtb = traffic_light(
        breaches_99=breaches.sum(),
        breaches_975=breaches.sum(),
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

# ── Tab 4: Stress Tests ──────────────────────────────────────────────────────

with tab4:
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
