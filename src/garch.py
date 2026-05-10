"""GARCH volatility modeling for VaR computation."""

from dataclasses import dataclass
import warnings

import numpy as np
from arch import arch_model


__all__ = ["GarchResult", "fit_garch", "forecast_vol"]


@dataclass
class GarchResult:
    """Container for GARCH model estimation results.

    Attributes:
        params: Dictionary of fitted model parameters.
        cond_vol: Array of conditional volatility estimates.
        forecasts: Array of volatility forecasts.
        aic: Akaike Information Criterion.
        bic: Bayesian Information Criterion.
    """

    params: dict
    cond_vol: np.ndarray
    forecasts: np.ndarray
    aic: float
    bic: float


def fit_garch(
    returns: np.ndarray,
    p: int = 1,
    q: int = 1,
    mean: str = "constant",
    dist: str = "normal",
) -> GarchResult:
    """Fit a GARCH(p,q) model and return conditional volatility.

    Parameters
    ----------
    returns : array-like of float
        Log returns series.
    p : int, optional
        GARCH lag order (default 1).
    q : int, optional
        ARCH lag order (default 1).
    mean : str, optional
        Mean model, one of "constant", "zero", "AR" (default "constant").
    dist : str, optional
        Error distribution, one of "normal", "t", "skewt" (default "normal").

    Returns
    -------
    GarchResult
        Fitted model parameters and conditional volatility.

    Raises
    ------
    ValueError
        If the series is too short (<100 observations).
    """
    returns = np.asarray(returns, dtype=float)
    if len(returns) < 100:
        raise ValueError(
            f"Need at least 100 observations, got {len(returns)}"
        )

    model = arch_model(returns, mean=mean, vol="GARCH", p=p, q=q, dist=dist)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fit = model.fit(disp="off")

    params = {k: v for k, v in fit.params.items()}
    aic = float(fit.aic)
    bic = float(fit.bic)

    return GarchResult(
        params=params,
        cond_vol=fit.conditional_volatility,
        forecasts=np.array([]),
        aic=aic,
        bic=bic,
    )


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
