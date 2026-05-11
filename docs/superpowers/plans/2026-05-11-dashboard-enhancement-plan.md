# Dashboard Enhancement Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Transform the VaR/ES dashboard from 4 bare-number tabs into a 7-tab professional portfolio piece with model explanation, methodology education, and rich visualizations — all aligned with the Swedish equity analysis from notebooks.

**Architecture:** Single-file Streamlit app (`app/streamlit_app.py`). Shared compute runs once before tabs (GARCH grid search, multi-alpha VaR/ES). Each tab reads from `st.session_state`. Educational content is static text extracted from notebook findings. No `src/` changes, no new dependencies.

**Tech Stack:** Python 3, Streamlit, Plotly Graph Objects, NumPy, Pandas, SciPy, existing `src/` modules

---

### Task 1: Foundation — Asset Universe, Multi-Alpha Compute, Helper Functions

**Files:**
- Modify: `app/streamlit_app.py` (lines 1-71, the setup + sidebar + compute section)

This task replaces the ticker list with the Swedish equity universe from the notebooks, upgrades shared compute to run GARCH grid search and multi-alpha VaR/ES, and adds all helper plot functions that tabs will use.

- [ ] **Step 1: Replace asset universe and add NAME_MAP constant**

Replace lines 28-31 (the `st.selectbox` for ticker). Add constants at module level after imports.

```python
# ── Constants: Swedish equity universe (aligned with notebooks) ──────────

ASSETS = ["^OMX", "ERIC-B.ST", "VOLV-B.ST", "HM-B.ST", "SWED-A.ST"]
NAMES  = ["OMXS30", "Ericsson", "Volvo", "H&M", "Swedbank"]
SECTORS = ["Broad Market Index", "Telecom Equipment", "Industrial / Automotive",
           "Consumer Retail", "Financial / Banking"]
NAME_MAP = dict(zip(ASSETS, NAMES))
SECTOR_MAP = dict(zip(ASSETS, SECTORS))
ALPHAS = [0.95, 0.975, 0.99]
```

- [ ] **Step 2: Update sidebar ticker selectbox to use display names**

Replace:
```python
ticker = st.selectbox(
    "Asset", ["^OMX", "^GSPC", "AAPL", "MSFT", "EURSEK=X"]
)
```

With:
```python
ticker = st.selectbox(
    "Asset",
    ASSETS,
    format_func=lambda x: f"{NAME_MAP[x]} ({x})  — {SECTOR_MAP[x]}"
)
```

- [ ] **Step 3: Add multi-alpha compute and grid search in the shared compute block**

Replace the try/except block starting at line 55. The shared compute block now runs `fit_garch_grid()` and computes VaR/ES at all 3 alpha levels.

```python
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
```

- [ ] **Step 4: Add helper plot functions after the load_data function and before sidebar**

Insert this block at line 51 (after `load_data` definition and before `with st.sidebar:`):

```python
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


def _plot_conditional_vol(returns_series, garch_result, ticker_name):
    """Conditional volatility time series with crisis annotations."""
    fig = go.Figure()
    date_index = pd.date_range(end=pd.Timestamp.today(), periods=len(garch_result.cond_vol), freq="B")
    fig.add_trace(go.Scatter(
        x=date_index[-len(garch_result.cond_vol):],
        y=garch_result.cond_vol,
        name="Conditional Volatility",
        line=dict(color="steelblue", width=1.5)
    ))
    # COVID-19 shaded region
    fig.add_vrect(
        x0="2020-02-19", x1="2020-03-23",
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
    fig.add_vrect(
        x0=0, x1=len(ret_arr),
        fillcolor="red", opacity=0.05,
        layer="below"
    )
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
```

- [ ] **Step 5: Run backend tests to confirm no breakage**

```bash
cd "/home/wd/Working Folder/Development/var-es" && python -m pytest tests/ -v --tb=short
```

Expected: all existing tests pass (no `src/` changes, helper functions are pure additions).

- [ ] **Step 6: Commit**

```bash
git add app/streamlit_app.py
git commit -m "feat(dashboard): add asset universe alignment and shared compute upgrade

Replace US/international tickers with 5 Swedish equities from notebooks.
Add multi-alpha VaR/ES compute, GARCH grid search on load, and helper
plot functions for all new visualizations."
```

---

### Task 2: Tab 1 (Executive Summary) + Tab 2 (Risk Snapshot Enhancement)

**Files:**
- Modify: `app/streamlit_app.py` (tabs section, lines 74-121)

- [ ] **Step 1: Replace the tab definition to include all 7 tabs**

Replace line 76:
```python
tab1, tab2, tab3, tab4 = st.tabs(
    ["Risk Snapshot", "Method Comparison", "Backtesting", "Stress Tests"]
)
```

With:
```python
( tab_exec, tab_snapshot, tab_compare, tab_model,
  tab_methods, tab_backtest, tab_stress ) = st.tabs([
    "Executive Summary", "Risk Snapshot", "Method Comparison",
    "Model Deep-Dive", "Methodology", "Backtesting", "Stress Tests"
])
```

- [ ] **Step 2: Implement Tab 1 — Executive Summary**

Replace the entire `with tab1:` block (lines 81-121). Insert this new Tab 1 content before the old Tab 1 (which becomes Tab 2).

```python
# ── Tab 1: Executive Summary ──────────────────────────────────────────────

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
        "Best GARCH Model", f"{garch_result.vol}({garch_result.p},{garch_result.q})-{garch_result.dist}" if use_garch and garch_result else "N/A",
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
```

- [ ] **Step 3: Enhance Tab 2 — Risk Snapshot**

Keep the existing `with tab1:` content (histogram + VaR/ES overlay). Add GARCH model card and educational expanders. After the existing `st.caption()` line (line 121), append:

```python
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
```

- [ ] **Step 4: Run backend tests to confirm no breakage**

```bash
cd "/home/wd/Working Folder/Development/var-es" && python -m pytest tests/ -v --tb=short
```

Expected: all existing tests pass.

- [ ] **Step 5: Commit**

```bash
git add app/streamlit_app.py
git commit -m "feat(dashboard): add executive summary and enhance risk snapshot

Add Tab 1 — Executive Summary with pipeline overview and spotlight metrics.
Enhance Tab 2 — Risk Snapshot with GARCH model card, half-life calculation,
and educational chart interpretation expanders."
```

---

### Task 3: Tab 3 (Method Comparison Enhancement) + Tab 4 (Model Deep-Dive)

**Files:**
- Modify: `app/streamlit_app.py` (tabs 2 and 3 sections, lines 123-176)

- [ ] **Step 1: Enhance Tab 3 — Method Comparison**

Keep existing comparison table + bar chart. After the existing `st.plotly_chart(fig, ...)` (line 175), add:

```python
    st.divider()

    # Distribution overlay for current selection
    st.subheader("Distribution Fit Analysis")
    dist_fig = _plot_distribution_overlay(returns, result, alpha)
    st.plotly_chart(dist_fig, use_container_width=True)

    with st.expander("Why Distribution Choice Matters"):
        st.markdown(f"""
        **Normal distribution underestimates tail risk.** The empirical
        returns have excess kurtosis (fat tails) — extreme events occur
        more frequently than the Normal distribution predicts.

        At {alpha:.1%} confidence:
        - **Parametric-Normal VaR:** {alpha_results[alpha].var:.4%}
        """)
        # Find parametric-t result if available
        try:
            r_t = alpha_results[alpha]
            st.markdown(f"- **Parametric-t VaR:** use Method Comparison table above for t-based estimate")
        except Exception:
            pass
        st.markdown("""
        - **Historical VaR:** Non-parametric — directly uses empirical
          percentiles, no distribution assumption needed

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
```

- [ ] **Step 2: Implement Tab 4 — Model Deep-Dive (new tab)**

Insert after the Tab 3 `with tab_compare:` block:

```python
# ── Tab 4: Model Deep-Dive ─────────────────────────────────────────────────

with tab_model:
    st.header("GARCH Model Deep-Dive")

    if not use_garch:
        st.info("Enable 'Use GARCH volatility' in the sidebar to see model diagnostics.")
    elif garch_grid_result is None:
        st.warning("GARCH grid search did not converge for this asset. Try a different ticker or date range.")
    else:
        gr = garch_grid_result

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

        # Grid search result summary
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

        # Persistence and half-life
        alpha_keys = [k for k in gr.params if 'alpha' in k]
        beta_keys = [k for k in gr.params if 'beta' in k]
        persistence = sum(abs(gr.params[k]) for k in alpha_keys + beta_keys)
        half_life = np.log(0.5) / np.log(max(persistence, 0.001)) if 0 < persistence < 1 else float('inf')

        col1, col2 = st.columns(2)
        col1.metric("Volatility Persistence", f"{persistence:.4f}",
                     delta="Stationary" if persistence < 1 else "Non-stationary")
        col2.metric("Half-Life", f"{half_life:.0f} trading days" if half_life < 1e6 else "∞",
                     delta=f"~{half_life/252:.1f} years" if half_life < 1e6 else None)

        # Conditional volatility plot
        st.subheader("Conditional Volatility")
        st.markdown(
            "**Context:** The GARCH model decomposes returns into a "
            "time-varying volatility component. Spikes correspond to "
            "market stress events."
        )
        try:
            vol_fig = _plot_conditional_vol(returns, gr, NAME_MAP[ticker])
            st.plotly_chart(vol_fig, use_container_width=True)
        except Exception as e:
            st.warning(f"Could not render conditional volatility plot: {e}")

        # Standardized residuals
        st.subheader("Model Diagnostics: Standardized Residuals")
        if hasattr(gr, 'cond_vol') and len(gr.cond_vol) > 1:
            std_resid = returns[-len(gr.cond_vol):] / np.maximum(gr.cond_vol, 1e-10)

            col1, col2 = st.columns(2)
            with col1:
                qq_fig = _plot_qq_residuals(std_resid, gr.dist, NAME_MAP[ticker])
                st.plotly_chart(qq_fig, use_container_width=True)

            with col2:
                # ACF of squared standardized residuals
                from statsmodels.tsa.stattools import acf
                sq_resid = std_resid ** 2
                acf_vals = acf(sq_resid, nlags=20)
                acf_fig = go.Figure()
                acf_fig.add_trace(go.Bar(
                    x=list(range(1, 21)), y=acf_vals[1:],
                    marker_color="steelblue"
                ))
                acf_fig.add_hline(
                    y=1.96/np.sqrt(len(sq_resid)),
                    line_dash="dash", line_color="red",
                    annotation_text="95% CI"
                )
                acf_fig.add_hline(
                    y=-1.96/np.sqrt(len(sq_resid)),
                    line_dash="dash", line_color="red"
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

        # EGARCH leverage explainer (if applicable)
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
                - This parameter is what distinguishes EGARCH from
                  standard GARCH — it explicitly models the empirical
                  observation that volatility rises more after market
                  declines than after equivalent gains

                **Finding (from notebooks):** EGARCH was selected for
                all five Swedish equities by AICc. The leverage effect
                is a systematic feature of this equity universe —
                negative shocks amplify future volatility 2-3x more
                than positive shocks of equal magnitude.
                """)

        st.info(
            "**Notebook reference:** Full grid search details and "
            "per-asset diagnostics in Notebook 02 — GARCH Volatility "
            "Modeling (`notebooks/02_garch_volatility.ipynb`)."
        )
```

- [ ] **Step 3: Run backend tests + verify Python syntax**

```bash
cd "/home/wd/Working Folder/Development/var-es" && python -c "import ast; ast.parse(open('app/streamlit_app.py').read()); print('Syntax OK')" && python -m pytest tests/ -v --tb=short
```

Expected: syntax OK + all tests pass.

- [ ] **Step 4: Commit**

```bash
git add app/streamlit_app.py
git commit -m "feat(dashboard): enhance method comparison and add model deep-dive

Tab 3: Add distribution fit overlay, confidence level sensitivity chart,
and method trade-off explanation.
Tab 4 (new): GARCH grid search results, conditional volatility plot,
standardized residual diagnostics (QQ + ACF), half-life calculation,
and EGARCH leverage effect explainer."
```

---

### Task 4: Tab 5 (Methodology) + Tab 6 (Backtesting Enhancement)

**Files:**
- Modify: `app/streamlit_app.py` (tabs 5 and 6 sections)

- [ ] **Step 1: Implement Tab 5 — Methodology (new tab)**

Insert after Tab 4's `with tab_model:` block:

```python
# ── Tab 5: Methodology ─────────────────────────────────────────────────────

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
        "Criterion": ["Accuracy (fat tails)", "Computation speed", "Regulatory acceptance",
                       "Forward-looking", "Requires distribution assumption"],
        "Historical": ["★★★★★", "★★★★★", "★★★★☆", "✗", "✗"],
        "Parametric-t": ["★★★★☆", "★★★★★", "★★★★☆", "✗", "✓"],
        "Parametric-Normal": ["★★☆☆☆", "★★★★★", "★★☆☆☆", "✗", "✓"],
        "Monte Carlo": ["★★★☆☆", "★★☆☆☆", "★★★☆☆", "✓", "✓"],
    })
    st.dataframe(guide_data, use_container_width=True, hide_index=True)
```

- [ ] **Step 2: Enhance Tab 6 — Backtesting**

Keep existing Tab 3 content (lines 177-268 of original file). When mapped to new tabs, this is Tab 6. Replace the existing `with tab3:` (now `with tab_backtest:`) content.

The backtesting code stays mostly the same — it already does rolling VaR, Kupiec, Christoffersen, traffic light, and breach plot. Keep all existing compute and just add educational expanders and new visualizations after the existing `st.plotly_chart(fig, ...)` in the backtesting section.

After the existing breach plot, add:

```python

    # Test interpretation cards
    st.subheader("Test Interpretation")
    with st.expander("What These Tests Actually Mean"):
        st.markdown(f"""
        **Kupiec POF Test** (p = {k_test.p_value:.4f}):
        Tests whether the number of VaR breaches matches expectation.
        - H₀: breach rate = expected rate ({1-alpha:.1%})
        - p > 0.05 → **pass** — breaches are at acceptable frequency
        - p < 0.05 → **fail** — model is miscalibrated
        - Result: **{'PASS ✓' if not k_test.reject else 'FAIL ✗'}**

        **Christoffersen Test** (p = {c_test.p_value:.4f}):
        Tests whether breaches are independent (not clustered).
        - H₀: breaches are independent over time
        - p > 0.05 → **pass** — breaches are randomly scattered,
          not clustered (good)
        - p < 0.05 → **fail** — breaches cluster together,
          model slow to adapt to regime changes
        - Result: **{'PASS ✓' if not c_test.reject else 'FAIL ✗'}**

        **Acerbi-Szekely Z2 Test** (p = {z2_test.p_value:.4f}):
        Tests whether ES forecasts are well-specified.
        - H₀: ES forecasts correctly capture tail severity
        - p > 0.05 → **pass** — ES estimates are adequate
        - p < 0.05 → **fail** — ES is miscalibrated (either too
          optimistic or too conservative)
        - Result: **{'PASS ✓' if not z2_test.reject else 'FAIL ✗'}**
        """)

    # FRTB dual-condition breach map
    st.subheader("FRTB Dual-Condition Breach Map")
    breach_fig = _plot_breach_timeline(ret_arr, breaches, breaches_975)
    st.plotly_chart(breach_fig, use_container_width=True)

    st.caption(
        "FRTB (2019) requires backtesting at both 99% and 97.5% "
        "confidence: green zone if ≤12 breaches at 99% AND ≤30 at "
        "97.5% over 250 days. Red crosses = 99% breaches. Orange "
        "triangles = 97.5% breaches only."
    )

    # Traffic light summary with regulatory context
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
            "Yellow zone: k=3.4–3.85 (capital add-on 13–28%). "
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
            "≤12 breaches at 99% AND ≤30 breaches at 97.5% (over 250 days). "
            "More stringent than Basel 1996 — single condition failure "
            "triggers red zone."
        )
```

- [ ] **Step 3: Run backend tests + syntax check**

```bash
cd "/home/wd/Working Folder/Development/var-es" && python -c "import ast; ast.parse(open('app/streamlit_app.py').read()); print('Syntax OK')" && python -m pytest tests/ -v --tb=short
```

- [ ] **Step 4: Commit**

```bash
git add app/streamlit_app.py
git commit -m "feat(dashboard): add methodology tab and enhance backtesting

Tab 5 (new): VaR/ES methodology education — how each method works,
distribution assumptions, ES coherence proof, method selection guide.
Tab 6: Add test interpretation cards, FRTB dual-condition breach map,
and regulatory capital multiplier explanation."
```

---

### Task 5: Tab 7 (Stress Tests Enhancement) + Sidebar Contextual Controls + Final Polish

**Files:**
- Modify: `app/streamlit_app.py` (tab 7 section, sidebar section for contextual controls)

- [ ] **Step 1: Enhance Tab 7 — Stress Tests**

Keep the existing stress test compute and scenario table. After the stress scenario bar chart, add:

```python

    st.divider()

    # Scenario waterfall chart
    st.subheader("Stress Escalation Waterfall")
    try:
        baseline_va = abs(alpha_results[0.975].var)
        waterfall_fig = _plot_scenario_waterfall(scenarios, baseline_va)
        st.plotly_chart(waterfall_fig, use_container_width=True)
    except Exception as e:
        st.warning(f"Could not render waterfall chart: {e}")

    st.caption(
        "The waterfall shows how VaR magnitude escalates from baseline "
        "(full-sample) → crisis scenarios → worst historical window. "
        "Each step represents the additional risk revealed by stress "
        "testing beyond normal-market VaR."
    )

    # Regulatory capital implication
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

    # Sensitivity analysis summary
    st.subheader("Scenario Severity Ranking")
    if scenarios:
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
```

- [ ] **Step 2: Add per-tab contextual sidebar controls**

At the end of the sidebar block (after the existing controls), add:

```python
    st.divider()

    # Contextual controls based on active tab
    # Use a session_state key to track which tab-specific controls to show.
    # Streamlit doesn't natively expose active tab, so we add a radio
    # that mirrors the tab selection for sidebar context.

    tab_context = st.radio(
        "Tab Context",
        ["Executive Summary", "Risk Snapshot", "Method Comparison",
         "Model Deep-Dive", "Methodology", "Backtesting", "Stress Tests"],
        key="sidebar_tab_context",
        label_visibility="collapsed"
    )

    if tab_context == "Model Deep-Dive":
        st.caption("Model Deep-Dive Controls")
        show_acf = st.checkbox("Show ACF diagnostics", value=True, key="show_acf")
        show_qq = st.checkbox("Show QQ plot", value=True, key="show_qq")
        st.session_state["show_acf"] = show_acf
        st.session_state["show_qq"] = show_qq

    elif tab_context == "Methodology":
        st.caption("Methodology Controls")
        selected_dist = st.selectbox(
            "Compare Distribution",
            ["Normal", "Student-t", "Both"],
            key="method_dist_select"
        )
        st.session_state["method_dist_select"] = selected_dist
```

- [ ] **Step 3: Full syntax check + run existing tests**

```bash
cd "/home/wd/Working Folder/Development/var-es" && python -c "
import ast
with open('app/streamlit_app.py') as f:
    ast.parse(f.read())
print('Syntax OK')
" && python -m pytest tests/ -v --tb=short
```

- [ ] **Step 4: Manual smoke test — verify Streamlit can import the app**

```bash
cd "/home/wd/Working Folder/Development/var-es" && streamlit run app/streamlit_app.py --server.headless true &
sleep 5
# Check if the process started
if pgrep -f streamlit; then
    echo "Streamlit started OK"
    pkill -f "streamlit run"
else
    echo "ERROR: Streamlit failed to start"
    exit 1
fi
```

- [ ] **Step 5: Run full backend test suite**

```bash
cd "/home/wd/Working Folder/Development/var-es" && python -m pytest tests/ -v --tb=short --cov=src --cov-report=term-missing
```

Expected: all tests pass, coverage ≥ 80%.

- [ ] **Step 6: Commit**

```bash
git add app/streamlit_app.py
git commit -m "feat(dashboard): enhance stress tests and add contextual sidebar

Tab 7: Add scenario waterfall chart, regulatory capital explanation,
and severity ranking table.
Sidebar: Add per-tab contextual controls for Model Deep-Dive and
Methodology tabs."
```

---

## Final Verification

After all tasks complete:

```bash
# Full test suite
cd "/home/wd/Working Folder/Development/var-es" && python -m pytest tests/ -v --tb=short --cov=src --cov-report=term-missing

# Verify app structure
python -c "
import ast
tree = ast.parse(open('app/streamlit_app.py').read())
# Count top-level definitions
funcs = [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)]
print(f'Functions: {len(funcs)}')
# Verify imports
imports = [n for n in ast.walk(tree) if isinstance(n, (ast.Import, ast.ImportFrom))]
print(f'Import statements: {len(imports)}')
print('Structure valid')
"

# Verify asset universe matches notebooks
python -c "
import ast, json
tree = ast.parse(open('app/streamlit_app.py').read())
# Find ASSETS constant
for node in ast.walk(tree):
    if isinstance(node, ast.Assign) and hasattr(node.targets[0], 'id') and node.targets[0].id == 'ASSETS':
        assets = [e.value for e in node.value.elts]
        expected = ['^OMX', 'ERIC-B.ST', 'VOLV-B.ST', 'HM-B.ST', 'SWED-A.ST']
        assert assets == expected, f'ASSETS mismatch: {assets} != {expected}'
        print('ASSET UNIVERSE: MATCHES NOTEBOOKS')
        break
"

# Confirm no src/ files were modified
git diff --name-only HEAD~5..HEAD | grep -v app/ | grep -v docs/ | grep -v .gitignore
# Expected: empty (no src/ changes)
```
