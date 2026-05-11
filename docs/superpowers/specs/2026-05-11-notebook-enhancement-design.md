# Notebook Enhancement Design

**Date:** 2026-05-11
**Branch:** phase/6-analysis-notebooks
**Scope:** All 5 analysis notebooks (01–05)

## Objective

Enhance 5 Jupyter notebooks with comprehensive explanation to serve dual purpose:
1. Deep learning tool for understanding risk analytics methodology
2. Professional portfolio piece showcasing capability to recruiters

## Current State

Notebooks have strong code, good theoretical background sections, validation checks,
and references. Gaps:
- Implementation sections lack narrative: code cells jump directly into execution
  without context
- Results lack interpretation: tables/plots printed with no guided reading
- Big-picture thread weak: sections read as independent blocks, not connected
  analytical journey
- Warning messages leak from `arch` library (GARCH optimization convergence)

## Design

### Cell-Level Template

Every code cell bracketed by two markdown cells:

**Before (Context markdown cell):**
```
### <Step Name>

**Purpose:** <1 sentence — what this cell does>

**Method:** <why this approach, statistical/computational choice, justification>

**Expected output:** <what we anticipate if model/data is well-behaved>
```

**After (Interpretation markdown cell):**
```
**Findings:** <2-4 bullets interpreting actual output>

**Connection:** <1 sentence linking to theory or next step>
```

Setup cells (imports, data loading): lighter version — Purpose only.

### Notebook-Level Structure

Each notebook follows this skeleton:

```
# 0X — Title
**Phase:** <position in pipeline>
**Prerequisites:** Notebook 0Y, modules src.X
**Learning Objectives:** <3-5 bullets — what reader will understand>

## 1. Motivation          ← enhance existing
## 2. Theoretical Background  ← enhance existing
## 3. Implementation      ← MAIN GAP: opening narrative + Context→Code→Interpretation
## 4. Results             ← MAIN GAP: interpretation after each table/figure
## 5. Validation          ← add pre/post for each check
## 6. Key Takeaways       ← strengthen narrative
## 7. References          ← existing
**Next: Notebook 0Z**    ← bridge to next in pipeline
```

### New Elements Per Notebook

- **Learning Objectives** block at top (3-5 bullets)
- **Prerequisites** line with explicit dependencies
- **Narrative opening** to Implementation section — what we're building, why this order
- **Narrative opening** to Results section — what theory predicts, how to read the evidence
- **Next notebook bridge** at end — "Next: Notebook 0Y covers..."

### Warning Suppression

Two sources:
1. Python `warnings` — already handled via `warnings.filterwarnings("ignore")`
2. `arch` library convergence warnings from `src/garch.py:89` — print directly via
   logger, not Python warnings system. Fix: add at top of notebooks that call GARCH:
   ```python
   import logging
   logging.getLogger("arch").setLevel(logging.ERROR)
   ```

No source code changes in `src/`. Notebook-only fix.

### Cross-Notebook References

Explicit back-references: "Recall in Notebook 01 we established fat tails (excess
kurtosis 5.5 for OMXS30) — we now model that volatility dynamics with EGARCH."
Forward bridges: "This conclusion motivates the GARCH modeling in Notebook 02."

### Execution Order

01 → 02 → 03 → 04 → 05 (pipeline order). Each builds on previous. Cross-refs only
point backward.

### Verification Per Notebook

1. Restart kernel, run all cells
2. Confirm: zero warnings in output
3. Confirm: every code cell has surrounding markdown (before + after or one of them)
4. Confirm: narrative readable start to finish without prior domain knowledge

## What Does NOT Change

- Code cells — untouched
- Existing markdown content — enhanced, not replaced
- Section numbering or overall structure
- Plot styling, colors, figure dimensions
- `src/` modules — no changes needed

## Notebooks in Scope

| # | Notebook | Key Enhancement Focus |
|---|----------|----------------------|
| 01 | Data Exploration & Stylised Facts | Already strong. Add cell-level context + interpretation. Tighten narrative. |
| 02 | GARCH Volatility | Thinnest on explanation. Heavy Implementation/Results enhancement needed. |
| 03 | VaR Methods & ES | Add cell context. Strengthen Results interpretation (coherence demo, portfolio). |
| 04 | Backtesting | Add cell context. Enhance FRTB section narrative. Interpret test outputs. |
| 05 | Stress Testing | Add cell context. Interpret scenario results. Connect to regulatory context. |
