from dataclasses import dataclass
import numpy as np


@dataclass
class VaRResult:
    var: float
    es: float
    method: str
    alpha: float
    horizon: int
    garch_used: bool


def compute_var_es(returns, method, alpha, horizon=1, garch_result=None):
    """Compute VaR and Expected Shortfall.

    Args:
        returns: array-like of log returns.
        method: "historical", "parametric", or "mc".
        alpha: confidence level (e.g., 0.975 for 97.5%).
        horizon: forecast horizon in days (default 1).
        garch_result: optional GarchResult for volatility scaling.

    Returns:
        VaRResult with var, es, and metadata.
    """
    returns = np.asarray(returns, dtype=float)

    if alpha <= 0 or alpha >= 1:
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")
    if horizon < 1:
        raise ValueError(f"horizon must be >= 1, got {horizon}")
    if method not in ("historical", "parametric", "mc"):
        raise ValueError(f"Unknown method: {method}")
    if len(returns) < 2:
        raise ValueError("Need at least 2 observations")

    if method == "historical":
        return _historical_var_es(returns, alpha, horizon, garch_result)
    else:
        raise NotImplementedError(f"Method {method} not yet implemented")


def _historical_var_es(returns, alpha, horizon, garch_result):
    var_1d = np.percentile(returns, (1 - alpha) * 100)
    tail = returns[returns <= var_1d]
    es_1d = tail.mean() if len(tail) > 0 else var_1d

    vol_scale = np.sqrt(horizon)
    if garch_result is not None and len(garch_result.cond_vol) > 0:
        vol_scale = garch_result.cond_vol[-1] * np.sqrt(horizon)

    return VaRResult(
        var=float(var_1d * vol_scale),
        es=float(es_1d * vol_scale),
        method="historical",
        alpha=alpha,
        horizon=horizon,
        garch_used=garch_result is not None,
    )
