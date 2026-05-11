"""Tests for app/charts.py chart functions."""

import numpy as np
from app.charts import plot_news_impact_curve
from plotly.graph_objects import Figure
from src.garch import GarchResult


def make_garch_result(vol="EGARCH", dist="t"):
    """Build a minimal GarchResult with known parameters."""
    return GarchResult(
        params={
            "omega": 0.1,
            "alpha[1]": 0.15,
            "beta[1]": 0.80,
            "gamma[1]": -0.10,
        },
        cond_vol=np.full(100, 0.02),
        forecasts=np.array([]),
        aic=-1200.0,
        aicc=-1199.0,
        bic=-1180.0,
        p=1, q=1,
        vol=vol, dist=dist,
    )


class TestNewsImpactCurve:

    def test_egarch_asymmetry(self):
        """EGARCH: negative shock has larger impact than positive of equal magnitude."""
        result = make_garch_result(vol="EGARCH")
        fig = plot_news_impact_curve(result)

        # Extract trace data
        trace = fig.data[0]
        eps = trace.x
        impacts = trace.y

        # Find impact at +2 and -2 (approximately)
        idx_pos = np.argmin(np.abs(np.array(eps) - 2.0))
        idx_neg = np.argmin(np.abs(np.array(eps) + 2.0))
        impact_pos = impacts[idx_pos]
        impact_neg = impacts[idx_neg]

        # For EGARCH with gamma<0: impact of -2 > impact of +2
        assert impact_neg > impact_pos, (
            f"Expected impact_neg ({impact_neg:.4f}) > impact_pos ({impact_pos:.4f})"
        )

        # Magnitude check: asymmetry should be substantial
        asymmetry = impact_neg - impact_pos
        assert asymmetry > 0.05, (
            f"Expected asymmetry > 0.05, got {asymmetry:.4f}"
        )

    def test_garch_symmetry(self):
        """GARCH: impact is symmetric (equal for +/- shocks)."""
        result = make_garch_result(vol="GARCH", dist="normal")
        fig = plot_news_impact_curve(result)

        trace = fig.data[0]
        eps = trace.x
        impacts = trace.y

        idx_pos = np.argmin(np.abs(np.array(eps) - 2.0))
        idx_neg = np.argmin(np.abs(np.array(eps) + 2.0))
        impact_pos = impacts[idx_pos]
        impact_neg = impacts[idx_neg]

        assert abs(impact_pos - impact_neg) < 1e-6, (
            f"Expected symmetric impact, got pos={impact_pos:.6f}, neg={impact_neg:.6f}"
        )

    def test_returns_figure(self):
        """Function returns a plotly Figure."""
        result = make_garch_result()
        fig = plot_news_impact_curve(result)
        assert isinstance(fig, Figure)

    def test_vline_at_zero(self):
        """Figure includes a vertical reference line at shock=0."""
        result = make_garch_result()
        fig = plot_news_impact_curve(result)
        vline_exists = any(
            hasattr(s, 'x0') and s.x0 == 0
            for s in (fig.layout.shapes or [])
        )
        assert vline_exists, "Expected vertical line at x=0"


    def test_yaxis_label_by_vol_type(self):
        """EGARCH uses log(sigma) label, GARCH uses sigma label."""
        result_egarch = make_garch_result(vol="EGARCH")
        fig_egarch = plot_news_impact_curve(result_egarch)
        assert "log(" in fig_egarch.layout.yaxis.title.text

        result_garch = make_garch_result(vol="GARCH", dist="normal")
        fig_garch = plot_news_impact_curve(result_garch)
        assert "log(" not in fig_garch.layout.yaxis.title.text

