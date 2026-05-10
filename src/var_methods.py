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


def compute_var_es(returns, method, alpha, horizon=1, garch_result=None, dist="normal", n_sim=10000):
    """Compute VaR and Expected Shortfall.

    Args:
        returns: array-like of log returns.
        method: "historical", "parametric", or "mc".
        alpha: confidence level (e.g., 0.975 for 97.5%).
        horizon: forecast horizon in days (default 1).
        garch_result: optional GarchResult for volatility scaling.
        dist: distribution for parametric method ("normal" or "t").
        n_sim: number of simulations for MC method (unused).

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
    elif method == "parametric":
        return _parametric_var_es(returns, alpha, horizon, garch_result, dist)
    elif method == "mc":
        raise NotImplementedError("MC not yet implemented")
    else:
        raise ValueError(f"Unknown method: {method}")


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


def _parametric_var_es(returns, alpha, horizon, garch_result, dist):
    from scipy import stats as sp_stats

    mu = np.mean(returns)
    sigma = np.std(returns, ddof=1)

    if garch_result is not None and len(garch_result.cond_vol) > 0:
        sigma = garch_result.cond_vol[-1]

    if dist == "normal":
        z_alpha = sp_stats.norm.ppf(1 - alpha)
        es_factor = sp_stats.norm.pdf(z_alpha) / (1 - alpha)
        var_1d = mu + z_alpha * sigma
        es_1d = mu - es_factor * sigma
    elif dist == "t":
        params = sp_stats.t.fit(returns)
        df, loc, scale = params
        if df < 2.1:
            df = 2.1
        t_alpha = sp_stats.t.ppf(1 - alpha, df)
        es_numerator = sp_stats.t.pdf(t_alpha, df)
        es_denom = 1 - alpha
        es_factor = (es_numerator / es_denom) * ((df + t_alpha**2) / (df - 1))
        var_1d = loc + t_alpha * scale
        es_1d = loc - es_factor * scale
    else:
        raise ValueError(f"Unknown distribution: {dist}")

    return VaRResult(
        var=float(var_1d * np.sqrt(horizon)),
        es=float(es_1d * np.sqrt(horizon)),
        method="parametric",
        alpha=alpha,
        horizon=horizon,
        garch_used=garch_result is not None,
    )
