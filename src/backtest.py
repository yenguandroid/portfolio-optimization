"""
backtest.py

Task 5: strategy backtesting utilities — simulate a portfolio's cumulative
value over a historical hold-out period (buy-and-hold or monthly
rebalancing back to target weights), and compute the standard performance
metrics needed to compare a strategy against a benchmark.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

TRADING_DAYS_PER_YEAR = 252


def simulate_portfolio(
    returns_wide: pd.DataFrame,
    weights: pd.Series,
    initial_value: float = 1.0,
    rebalance: Optional[str] = None,
) -> pd.Series:
    """
    Simulate a portfolio's cumulative value over time given daily simple
    returns for each asset.

    Parameters
    ----------
    returns_wide : pd.DataFrame
        Daily simple returns, indexed by date, one column per asset (must
        contain a column for every ticker in `weights.index`). No NaNs
        (align/drop before calling).
    weights : pd.Series
        Target portfolio weights indexed by the same tickers as
        `returns_wide`'s columns, summing to 1.
    initial_value : float
        Starting portfolio value (e.g. 1.0 for a normalized cumulative
        return series, or 10000 for a dollar-value simulation).
    rebalance : {None, "M"}
        If None: buy-and-hold — weights drift with each asset's relative
        performance over the whole period (no rebalancing).
        If "M": rebalance back to target weights at the start of every
        calendar month.

    Returns
    -------
    pd.Series indexed by date, the portfolio's cumulative value each day.
    """
    assets = list(weights.index)
    missing = [a for a in assets if a not in returns_wide.columns]
    if missing:
        raise ValueError(f"returns_wide is missing columns for: {missing}")

    price_relatives = 1.0 + returns_wide[assets]
    weights = weights / weights.sum()  # defensive normalization

    asset_values = weights.values * initial_value
    portfolio_values = []
    current_period = None

    for date, row in price_relatives.iterrows():
        if rebalance == "M":
            period = date.to_period("M")
            if period != current_period:
                total = asset_values.sum()
                asset_values = weights.values * total
                current_period = period
        elif rebalance is not None:
            raise ValueError(f"Unsupported rebalance option: {rebalance}")

        asset_values = asset_values * row.values
        portfolio_values.append(asset_values.sum())

    return pd.Series(portfolio_values, index=price_relatives.index, name="portfolio_value")


def returns_from_value_series(value_series: pd.Series) -> pd.Series:
    """Daily simple returns implied by a cumulative value series."""
    return value_series.pct_change().dropna()


def max_drawdown(value_series: pd.Series) -> float:
    """
    Maximum drawdown: the largest peak-to-trough decline in the value
    series, expressed as a positive fraction (e.g. 0.35 == a 35% decline
    from the running peak at some point in the period).
    """
    running_max = value_series.cummax()
    drawdown = (value_series - running_max) / running_max
    return float(-drawdown.min())


def backtest_metrics(
    value_series: pd.Series,
    risk_free_rate: float = 0.02,
) -> dict:
    """
    Compute the standard Task 5 performance metrics for a simulated
    portfolio value series: total return, annualized return, Sharpe
    Ratio, and maximum drawdown.
    """
    daily_returns = returns_from_value_series(value_series)
    n_days = len(value_series) - 1

    total_return = float(value_series.iloc[-1] / value_series.iloc[0] - 1.0)
    years = n_days / TRADING_DAYS_PER_YEAR
    annualized_return = (1.0 + total_return) ** (1.0 / years) - 1.0 if years > 0 else np.nan

    daily_rf = (1 + risk_free_rate) ** (1 / TRADING_DAYS_PER_YEAR) - 1
    excess = daily_returns - daily_rf
    sharpe = (
        (excess.mean() / excess.std()) * np.sqrt(TRADING_DAYS_PER_YEAR)
        if not np.isclose(excess.std(), 0.0, atol=1e-12)
        else np.nan
    )

    mdd = max_drawdown(value_series)

    return {
        "Total Return": total_return,
        "Annualized Return": annualized_return,
        "Sharpe Ratio": float(sharpe),
        "Max Drawdown": mdd,
    }


def compare_backtest(results: dict) -> pd.DataFrame:
    """
    Build a side-by-side comparison table from a dict of
    {strategy_name: metrics_dict} (each metrics_dict from `backtest_metrics`).
    """
    table = pd.DataFrame(results).T
    table = table[["Total Return", "Annualized Return", "Sharpe Ratio", "Max Drawdown"]]
    table.index.name = "Portfolio"
    return table
