"""Data fetching and return computation utilities."""

import numpy as np
import pandas as pd
import yfinance as yf


def compute_returns(prices, method="log"):
    """Compute returns from price series.

    Parameters
    ----------
    prices : array-like
        Price series.
    method : str, optional
        "log" for log returns, "simple" for arithmetic returns.
        Default is "log".

    Returns
    -------
    np.ndarray
        Array of returns with length len(prices) - 1.

    Raises
    ------
    ValueError
        If method is not "log" or "simple".
    """
    prices = np.asarray(prices, dtype=float)
    if method == "log":
        return np.diff(np.log(prices))
    if method == "simple":
        return np.diff(prices) / prices[:-1]
    raise ValueError(f"Unknown method: {method}")


def fetch_prices(tickers, start, end):
    """Fetch adjusted close prices from yfinance.

    Parameters
    ----------
    tickers : str or list of str
        Ticker symbol(s) to fetch.
    start : str
        Start date in "YYYY-MM-DD" format.
    end : str
        End date in "YYYY-MM-DD" format.

    Returns
    -------
    pd.DataFrame
        DataFrame of close prices, one column per ticker.

    Raises
    ------
    ValueError
        If no data is returned for the given tickers.
    """
    if isinstance(tickers, str):
        tickers = [tickers]
    data = yf.download(tickers, start=start, end=end, progress=False)
    if data.empty:
        raise ValueError(f"No data returned for {tickers}")
    if "Adj Close" in data.columns:
        return data["Adj Close"]
    return data["Close"]
