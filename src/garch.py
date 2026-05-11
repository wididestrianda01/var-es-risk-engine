"""GARCH volatility modeling for VaR computation."""

from dataclasses import dataclass
import warnings

import numpy as np
from arch import arch_model


__all__ = ["GarchResult", "fit_garch", "fit_garch_grid", "forecast_vol"]


@dataclass
class GarchResult:
    """Container for GARCH model estimation results.

    Attributes:
        params: Dictionary of fitted model parameters.
        cond_vol: Array of conditional volatility estimates.
        forecasts: Array of volatility forecasts.
        aic: Akaike Information Criterion.
        aicc: Corrected AIC (AICc).
        bic: Bayesian Information Criterion.
        p: ARCH order.
        q: GARCH order.
        vol: Volatility model (GARCH, EGARCH).
        dist: Error distribution (normal, t).
    """

    params: dict
    cond_vol: np.ndarray
    forecasts: np.ndarray
    aic: float
    aicc: float
    bic: float
    p: int = 1
    q: int = 1
    vol: str = "GARCH"
    dist: str = "normal"


def fit_garch(
    returns: np.ndarray,
    p: int = 1,
    q: int = 1,
    mean: str = "constant",
    vol: str = "GARCH",
    dist: str = "normal",
) -> GarchResult:
    """Fit a GARCH/EGARCH(p,q) model and return conditional volatility.

    Parameters
    ----------
    returns : array-like of float
        Log returns series.
    p : int, optional
        ARCH lag order (default 1).
    q : int, optional
        GARCH lag order (default 1).
    mean : str, optional
        Mean model, one of "constant", "zero", "AR" (default "constant").
    vol : str, optional
        Volatility model: "GARCH" or "EGARCH" (default "GARCH").
    dist : str, optional
        Error distribution: "normal" or "t" (default "normal").

    Returns
    -------
    GarchResult
        Fitted model parameters and conditional volatility.

    Raises
    ------
    ValueError
        If the series is too short (<100 observations).
    """
    returns = np.asarray(returns, dtype=float).ravel()
    returns = returns[np.isfinite(returns)]
    if len(returns) < 100:
        raise ValueError(
            f"Need at least 100 observations, got {len(returns)}"
        )

    model = arch_model(returns, mean=mean, vol=vol, p=p, q=q, dist=dist)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        # Use BHHH-style optimizer with tighter tolerance for better convergence
        # on heavy-tailed equity return data
        fit = model.fit(
            disp="off",
            options={"maxiter": 2000, "ftol": 1e-12},
        )

    params = {k: v for k, v in fit.params.items()}
    aic = float(fit.aic)
    bic = float(fit.bic)
    n = len(returns)
    k = p + q + 2  # omega + alpha_i + beta_j + mu
    aicc = aic + 2 * k * (k + 1) / max(n - k - 1, 1)

    return GarchResult(
        params=params,
        cond_vol=fit.conditional_volatility,
        forecasts=np.array([]),
        aic=aic,
        aicc=aicc,
        bic=bic,
        p=p,
        q=q,
        vol=vol,
        dist=dist,
    )


def fit_garch_grid(
    returns: np.ndarray,
    max_p: int = 2,
    max_q: int = 2,
    mean: str = "constant",
    vol_specs: list | None = None,
    dist_specs: list | None = None,
) -> GarchResult:
    """Grid search over vol × dist × (p,q), selecting best by AICc.

    Parameters
    ----------
    returns : array-like of float
        Log returns series.
    max_p : int, optional
        Maximum ARCH lag to try (default 2).
    max_q : int, optional
        Maximum GARCH lag to try (default 2).
    mean : str, optional
        Mean model (default "constant").
    vol_specs : list of str, optional
        Volatility models to try. Default: ["GARCH", "EGARCH"].
    dist_specs : list of str, optional
        Distributions to try. Default: ["normal", "t"].

    Returns
    -------
    GarchResult
        Best model by AICc across all specifications.
    """
    if vol_specs is None:
        vol_specs = ["GARCH", "EGARCH"]
    if dist_specs is None:
        dist_specs = ["normal", "t"]

    returns = np.asarray(returns, dtype=float).ravel()
    returns = returns[np.isfinite(returns)]
    if len(returns) < 100:
        raise ValueError(
            f"Need at least 100 observations, got {len(returns)}"
        )

    best_result = None
    best_aicc = np.inf

    for vol in vol_specs:
        for dist in dist_specs:
            for p in range(1, max_p + 1):
                for q_val in range(1, max_q + 1):
                    try:
                        result = fit_garch(
                            returns, p=p, q=q_val,
                            mean=mean, vol=vol, dist=dist,
                        )
                        if result.aicc < best_aicc:
                            best_aicc = result.aicc
                            best_result = result
                    except Exception:
                        continue

    if best_result is None:
        raise RuntimeError(
            "No model converged across vol × dist × (p,q) grid"
        )

    return best_result


def forecast_vol(result: GarchResult, horizon: int = 1) -> np.ndarray:
    """Forecast volatility from a fitted GarchResult.

    Uses the square-root-of-time rule scaled from the last conditional
    volatility estimate.

    Parameters
    ----------
    result : GarchResult
        Fitted model result.
    horizon : int, optional
        Forecast horizon in days (default 1).

    Returns
    -------
    np.ndarray
        Volatility forecast for the given horizon.
    """
    if result.cond_vol is None or len(result.cond_vol) == 0:
        raise ValueError("No conditional volatility in result")
    last_vol = result.cond_vol[-1]
    return last_vol * np.sqrt(horizon)
