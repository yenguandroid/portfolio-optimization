"""
risk_metrics.py

Foundational risk metrics: historical & parametric Value at Risk (VaR)
and the (annualized) Sharpe Ratio.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import norm

TRADING_DAYS_PER_YEAR = 252


def historical_var(returns: pd.Series, confidence: float = 0.95) -> float:
    """
    Historical (empirical) VaR: the loss threshold not expected to be
    exceeded with the given confidence level, based on the empirical
    return distribution. Returned as a positive number representing a
    loss (e.g. 0.03 == a 3% loss).
    """
    clean = returns.dropna()
    var = -np.percentile(clean, (1 - confidence) * 100)
    return var


def parametric_var(returns: pd.Series, confidence: float = 0.95) -> float:
    """
    Parametric (variance-covariance) VaR assuming normally distributed
    returns. Returned as a positive number representing a loss.
    """
    clean = returns.dropna()
    mu, sigma = clean.mean(), clean.std()
    z = norm.ppf(1 - confidence)
    var = -(mu + z * sigma)
    return var


def sharpe_ratio(
    returns: pd.Series,
    risk_free_rate_annual: float = 0.02,
    periods_per_year: int = TRADING_DAYS_PER_YEAR,
) -> float:
    """
    Annualized Sharpe Ratio = (mean daily excess return / std daily return)
    * sqrt(periods_per_year).

    Parameters
    ----------
    returns : pd.Series
        Daily simple returns.
    risk_free_rate_annual : float
        Annual risk-free rate (default 2%), converted to a daily rate.
    """
    clean = returns.dropna()
    daily_rf = (1 + risk_free_rate_annual) ** (1 / periods_per_year) - 1
    excess_returns = clean - daily_rf
    std = excess_returns.std()
    if np.isclose(std, 0.0, atol=1e-12):
        return np.nan
    return (excess_returns.mean() / std) * np.sqrt(periods_per_year)


def summarize_risk(df: pd.DataFrame, confidence: float = 0.95, risk_free_rate_annual: float = 0.02) -> pd.DataFrame:
    """
    Build a summary table of annualized return, annualized volatility,
    historical VaR, parametric VaR, and Sharpe ratio per ticker.
    """
    rows = []
    for ticker, group in df.groupby("Ticker"):
        r = group["Daily_Return"].dropna()
        rows.append({
            "Ticker": ticker,
            "Annualized_Return": r.mean() * TRADING_DAYS_PER_YEAR,
            "Annualized_Volatility": r.std() * np.sqrt(TRADING_DAYS_PER_YEAR),
            f"Historical_VaR_{int(confidence*100)}": historical_var(r, confidence),
            f"Parametric_VaR_{int(confidence*100)}": parametric_var(r, confidence),
            "Sharpe_Ratio": sharpe_ratio(r, risk_free_rate_annual),
        })
    return pd.DataFrame(rows).set_index("Ticker")
