# Dashboard Enhancement Design

**Date:** 2026-05-11
**Branch:** phase/7-dashboard-enhancement
**Scope:** `app/streamlit_app.py` — dashboard-only changes, no `src/` modifications

## Objective

Transform the VaR/ES dashboard from a bare-numbers display into a professional portfolio piece that demonstrates end-to-end understanding of risk analytics methodology. Target audience: hiring managers and recruiters evaluating capability breadth and depth.

## Current State

Dashboard has 4 tabs (Risk Snapshot, Method Comparison, Backtesting, Stress Tests) showing VaR/ES numbers and basic plots. No educational content, no model explanation, no methodology narrative. User sees results without understanding why they matter or how they were produced.

## Target State

**7 flat tabs** (Streamlit-native `st.tabs()`), logical pipeline order:

| # | Tab | Status | Content |
|---|-----|--------|---------|
| 1 | Executive Summary | NEW | Pipeline overview, key capability highlights, 3 spotlight metrics |
| 2 | Risk Snapshot | ENHANCED | Current tab 1 + GARCH model card, expandable "How to read" panels |
| 3 | Method Comparison | ENHANCED | Current tab 2 + distribution overlay, method trade-off matrix |
| 4 | Model Deep-Dive | NEW | GARCH grid search results, conditional volatility, residual diagnostics, half-life comparison, EGARCH leverage explainer |
| 5 | Methodology | NEW | How each VaR method works, tail distribution comparison, ES coherence demonstration |
| 6 | Backtesting | ENHANCED | Current tab 3 + test interpretation cards, breach calendar heatmap, FRTB dual-condition visual |
| 7 | Stress Tests | ENHANCED | Current tab 4 + scenario waterfall, sensitivity tornado, regulatory capital summary |

## Visual Upgrades

### Model Diagnostics (Tab 4)
- Grid search results table (all 16 spec combinations ranked by AICc)
- Conditional volatility time series with crisis annotations (COVID, rate hike)
- Standardized residual QQ plots per asset
- Half-life comparison bar chart across assets
- EGARCH leverage coefficient visualization

### Tail Analysis (Tab 3, 5)
- Distribution overlay: normal PDF vs Student-t PDF vs empirical histogram with VaR annotation
- Method comparison across confidence levels (95/97.5/99%) — grouped bar chart
- ES decomposition waterfall (mean + tail multiplier = ES)

### Regulatory Context (Tab 6, 7)
- Traffic light heatmap (green/yellow/red) per framework
- Breach timeline with regime overlays (COVID shaded region)
- Scenario waterfall chart (baseline → crisis scenarios → worst window)
- Sensitivity tornado plot

## Educational Content Strategy

Three content types embedded per tab:

1. **Info callouts** (`st.info()`, `st.expander()`) — 2-3 sentence explainers of what the user is seeing and why it matters
2. **Metric captions** (`st.caption()`) — one-line significance notes under key numbers
3. **Methodology cards** — structured expandable sections with "What", "Why", "Finding" format, pulling directly from notebook Interpretation cells

Content is **static text** extracted from notebook findings — not dynamically generated from live analysis. Notebook analysis (e.g., "EGARCH-t selected for all 5 assets, leverage effect systematic") is hardcoded as educational narrative.

## Data Flow

```
Sidebar controls ──→ session_state
                         │
load_data() ──→ prices, returns  [@st.cache_data, TTL 3600]
                         │
              ┌──────────┼──────────┐
              │          │          │
         fit_garch()  fit_garch_grid()  compute_var_es()
              │          │          │     (×3 alpha levels)
              │          │          │
              └──────────┼──────────┘
                         │
              st.session_state (shared results)
                         │
     ┌───────────────────┼───────────────────┐
     │       │       │       │       │       │
   Tab 1  Tab 2  Tab 3  Tab 4  Tab 5  Tab 6  Tab 7
```

- Shared compute done once before tabs. No recomputation on tab switch.
- `fit_garch_grid()` runs on load to populate Model Deep-Dive tab
- VaR/ES computed at all 3 confidence levels for multi-alpha displays

## Key Design Decisions

1. **Single file.** No splitting `streamlit_app.py`. 7 tabs in one file with `st.tabs()`. Streamlit handles this cleanly.
2. **No new backend code.** All new content is UI layer. `src/` modules unchanged.
3. **No new dependencies.** All visualizations use existing `plotly.graph_objects`.
4. **Helper functions for plots.** Extract each new chart into a named function (`_plot_conditional_vol()`, `_plot_tail_comparison()`, etc.) to keep tab code readable.
5. **Static educational content.** Notebook findings are hardcoded narrative — not live analysis. Avoids recomputation overhead and keeps dashboard fast.
6. **Per-tab controls.** Sidebar shows contextual controls based on active tab via `st.session_state` (e.g., Model Deep-Dive shows asset selector, Methodology shows distribution selector).

## Error Handling

- **Top-level** (existing): data load failure → `st.error` + `st.stop()`
- **Per-tab** (new): plot generation failure → `st.warning("Could not render X: ...")` + continue rendering other content
- **Model Deep-Dive tab**: if `fit_garch_grid()` fails for an asset, skip that asset's diagnostics with a note, don't block the tab

## Interactivity (Moderate)

- Per-tab contextual controls in sidebar
- Expandable info panels (`st.expander()`) for methodology explanations
- Tooltip-like captions under metrics
- Collapsible methodology cards
- No custom JavaScript or complex state machines

## Testing

- Visual smoke tests: verify all 7 tabs render for each ticker
- Existing backend tests cover compute correctness (no `src/` changes)
- Edge cases: short date ranges, missing data, failed GARCH convergence — handled by existing patterns

## What Does NOT Change

- `src/` modules (garch.py, var_methods.py, backtest.py, stress_test.py, utils.py)
- `tests/` directory
- `notebooks/` directory
- Data pipeline (`data/` directory)
- Sidebar core controls (ticker, alpha, horizon, method, GARCH toggle, date range)

## Tab Detail Specifications

### Tab 1: Executive Summary
- Pipeline flow diagram (5-step: Data → GARCH → VaR/ES → Backtest → Stress) using Plotly or emoji-based layout
- 3 spotlight metrics: best model AICc, backtest green zone status, worst-case stress scenario
- Key capability summary bullets (what this pipeline demonstrates)

### Tab 2: Risk Snapshot
- Existing histogram + VaR/ES overlay (keep)
- **New**: GARCH model summary card (which spec selected, parameter count, AICc)
- **New**: "How to read this chart" expander
- **New**: ES > VaR coherence explainer caption

### Tab 3: Method Comparison
- Existing comparison table + bar chart (keep)
- **New**: Distribution fit overlay plot (normal vs t vs empirical)
- **New**: Method trade-off matrix (accuracy vs speed vs assumptions)
- **New**: Confidence level sensitivity chart

### Tab 4: Model Deep-Dive
- Grid search results table (vol × dist × (p,q) ranked)
- Conditional volatility time series plot(s)
- Standardized residual QQ plot
- Half-life comparison bar chart
- EGARCH leverage coefficient visualization (beta sign/magnitude)
- Each visualization preceded by context explainer

### Tab 5: Methodology
- Visual step-by-step for each method (historical → sort & percentile, parametric → fit distribution → invert CDF, MC → simulate paths → percentile)
- Tail distribution comparison plot
- ES coherence property explanation with example
- Method selection decision guide

### Tab 6: Backtesting
- Existing rolling backtest plot + metrics (keep)
- **New**: Test interpretation cards (what Kupiec p=0.80 actually means)
- **New**: Breach calendar heatmap (date × year grid)
- **New**: FRTB dual-condition visual (99% + 97.5% breaches on same timeline)

### Tab 7: Stress Tests
- Existing scenario table + bar chart (keep)
- **New**: Scenario waterfall chart
- **New**: Sensitivity tornado plot
- **New**: Regulatory capital implication summary
