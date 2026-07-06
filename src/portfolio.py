"""
portfolio.py

Task 4: Modern Portfolio Theory (MPT) utilities — combining a forecast-derived
expected return for TSLA with historical expected returns for BND/SPY, a
historical covariance matrix, and an efficient-frontier optimization built on
scipy.optimize (no PyPortfolioOpt dependency required, though that library is
an equally valid choice per the task instructions).
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
import pandas as pd
from scipy.optimize import minimize

TRADING_DAYS_PER_YEAR = 252


# ---------------------------------------------------------------------------
# Expected returns
# ---------------------------------------------------------------------------

def historical_annual_return(daily_returns: pd.Series) -> float:
    """Annualize a historical mean daily return: mean * 252 trading days."""
    return float(daily_returns.dropna().mean() * TRADING_DAYS_PER_YEAR)


def build_expected_returns(
    combined_df: pd.DataFrame,
    tsla_forecast_annual_return: float,
    forecast_ticker: str = "TSLA",
    return_col: str = "Daily_Return",
) -> pd.Series:
    """
    Build the expected-returns vector used by MPT: a forecast-derived
    expected return for TSLA, and historical-average annualized returns
    for every other ticker (BND, SPY) — the "analyst has a view on one
    asset, relies on history for the rest" scenario described in Task 4.

    Parameters
    ----------
    combined_df : pd.DataFrame
        Long-format frame with columns [Ticker, return_col] (e.g. the
        Task 1 cleaned dataset with Daily_Return already computed).
    tsla_forecast_annual_return : float
        The TSLA expected annual return implied by the Task 3 forecast
        (e.g. derived from the forecasted price path's implied CAGR over
        the forecast horizon — see notebook Section 2 for the exact
        derivation).
    forecast_ticker : str
        Which ticker the forecast-derived return applies to (default TSLA).
    return_col : str
        Column holding daily simple returns.

    Returns
    -------
    pd.Series indexed by ticker, annualized expected returns.
    """
    tickers = sorted(combined_df["Ticker"].unique())
    expected = {}
    for ticker in tickers:
        if ticker == forecast_ticker:
            expected[ticker] = tsla_forecast_annual_return
        else:
            ticker_returns = combined_df.loc[combined_df["Ticker"] == ticker, return_col]
            expected[ticker] = historical_annual_return(ticker_returns)
    return pd.Series(expected).sort_index()


def tsla_forecast_annual_return_from_prices(current_price: float, forecast_price: float, horizon_days: int) -> float:
    """
    Convert a Task 3 point-forecast price (n trading days out) into an
    implied annualized expected return (CAGR), for use as TSLA's expected
    return in `build_expected_returns`.

    Parameters
    ----------
    current_price : float
        The last known actual price (forecast origin).
    forecast_price : float
        The point forecast (e.g. `future_forecast["forecast"].iloc[-1]`
        from Task 3).
    horizon_days : int
        Number of trading days between current_price and forecast_price.
    """
    total_return = (forecast_price / current_price) - 1.0
    years = horizon_days / TRADING_DAYS_PER_YEAR
    annualized = (1.0 + total_return) ** (1.0 / years) - 1.0
    return float(annualized)


# ---------------------------------------------------------------------------
# Covariance matrix
# ---------------------------------------------------------------------------

def compute_covariance_matrix(combined_df: pd.DataFrame, return_col: str = "Daily_Return") -> pd.DataFrame:
    """
    Annualized covariance matrix of daily returns across tickers.

    Parameters
    ----------
    combined_df : pd.DataFrame
        Long-format frame with columns [Date, Ticker, return_col].

    Returns
    -------
    pd.DataFrame, tickers x tickers, annualized covariance (daily cov * 252).
    """
    wide = combined_df.pivot(index="Date", columns="Ticker", values=return_col)
    daily_cov = wide.cov()
    annual_cov = daily_cov * TRADING_DAYS_PER_YEAR
    return annual_cov.sort_index().sort_index(axis=1)


# ---------------------------------------------------------------------------
# Portfolio performance
# ---------------------------------------------------------------------------

def portfolio_performance(
    weights: np.ndarray,
    expected_returns: pd.Series,
    cov_matrix: pd.DataFrame,
) -> Tuple[float, float]:
    """Return (expected annual return, annual volatility) for given weights."""
    weights = np.asarray(weights)
    port_return = float(np.dot(weights, expected_returns.values))
    port_vol = float(np.sqrt(weights.T @ cov_matrix.values @ weights))
    return port_return, port_vol


def sharpe_ratio_for_weights(
    weights: np.ndarray,
    expected_returns: pd.Series,
    cov_matrix: pd.DataFrame,
    risk_free_rate: float = 0.02,
) -> float:
    port_return, port_vol = portfolio_performance(weights, expected_returns, cov_matrix)
    if port_vol == 0:
        return np.nan
    return (port_return - risk_free_rate) / port_vol


# ---------------------------------------------------------------------------
# Optimization: Max Sharpe (tangency) and Min Volatility portfolios
# ---------------------------------------------------------------------------

def _weight_bounds_and_constraints(n_assets: int, bounds=(0.0, 1.0)):
    constraints = ({"type": "eq", "fun": lambda w: np.sum(w) - 1.0},)
    bnds = tuple(bounds for _ in range(n_assets))
    initial_guess = np.repeat(1.0 / n_assets, n_assets)
    return bnds, constraints, initial_guess


def optimize_max_sharpe(
    expected_returns: pd.Series,
    cov_matrix: pd.DataFrame,
    risk_free_rate: float = 0.02,
    bounds: tuple = (0.0, 1.0),
) -> dict:
    """
    Find the long-only portfolio weights maximizing the Sharpe Ratio
    (the tangency portfolio).

    Returns
    -------
    dict with keys: weights (pd.Series), expected_return, volatility, sharpe_ratio
    """
    n = len(expected_returns)
    bnds, constraints, x0 = _weight_bounds_and_constraints(n, bounds)

    def neg_sharpe(w):
        return -sharpe_ratio_for_weights(w, expected_returns, cov_matrix, risk_free_rate)

    result = minimize(neg_sharpe, x0, method="SLSQP", bounds=bnds, constraints=constraints)
    if not result.success:
        raise RuntimeError(f"Max-Sharpe optimization failed: {result.message}")

    weights = pd.Series(result.x, index=expected_returns.index)
    ret, vol = portfolio_performance(result.x, expected_returns, cov_matrix)
    sharpe = (ret - risk_free_rate) / vol
    return {"weights": weights, "expected_return": ret, "volatility": vol, "sharpe_ratio": sharpe}


def optimize_min_volatility(
    expected_returns: pd.Series,
    cov_matrix: pd.DataFrame,
    risk_free_rate: float = 0.02,
    bounds: tuple = (0.0, 1.0),
) -> dict:
    """
    Find the long-only portfolio weights minimizing volatility, regardless
    of expected return.

    Returns
    -------
    dict with keys: weights (pd.Series), expected_return, volatility, sharpe_ratio
    """
    n = len(expected_returns)
    bnds, constraints, x0 = _weight_bounds_and_constraints(n, bounds)

    def vol_only(w):
        return portfolio_performance(w, expected_returns, cov_matrix)[1]

    result = minimize(vol_only, x0, method="SLSQP", bounds=bnds, constraints=constraints)
    if not result.success:
        raise RuntimeError(f"Min-volatility optimization failed: {result.message}")

    weights = pd.Series(result.x, index=expected_returns.index)
    ret, vol = portfolio_performance(result.x, expected_returns, cov_matrix)
    sharpe = (ret - risk_free_rate) / vol if vol > 0 else np.nan
    return {"weights": weights, "expected_return": ret, "volatility": vol, "sharpe_ratio": sharpe}


# ---------------------------------------------------------------------------
# Efficient frontier
# ---------------------------------------------------------------------------

def compute_efficient_frontier(
    expected_returns: pd.Series,
    cov_matrix: pd.DataFrame,
    n_points: int = 50,
    bounds: tuple = (0.0, 1.0),
) -> pd.DataFrame:
    """
    Trace the efficient frontier: for a range of target returns spanning
    the achievable range, find the minimum-volatility portfolio that
    achieves at least that return.

    Returns
    -------
    pd.DataFrame with columns [target_return, volatility, weights_<ticker>...]
    sorted by target_return, restricted to the frontier's efficient
    (upper) branch.
    """
    n = len(expected_returns)
    bnds, base_constraints, x0 = _weight_bounds_and_constraints(n, bounds)

    min_ret = expected_returns.min()
    max_ret = expected_returns.max()
    target_returns = np.linspace(min_ret, max_ret, n_points)

    rows = []
    for target in target_returns:
        constraints = base_constraints + (
            {"type": "eq", "fun": lambda w, target=target: np.dot(w, expected_returns.values) - target},
        )

        def vol_only(w):
            return portfolio_performance(w, expected_returns, cov_matrix)[1]

        result = minimize(vol_only, x0, method="SLSQP", bounds=bnds, constraints=constraints)
        if not result.success:
            continue

        ret, vol = portfolio_performance(result.x, expected_returns, cov_matrix)
        row = {"target_return": ret, "volatility": vol}
        for ticker, w in zip(expected_returns.index, result.x):
            row[f"weight_{ticker}"] = w
        rows.append(row)

    frontier = pd.DataFrame(rows).sort_values("volatility").reset_index(drop=True)

    # Keep only the efficient (upper) branch: for each volatility level,
    # only the highest-return portfolio is "efficient" in the MPT sense.
    frontier["running_max_return"] = frontier["target_return"].cummax()
    efficient = frontier[frontier["target_return"] >= frontier["running_max_return"] - 1e-9].copy()
    efficient = efficient.drop(columns="running_max_return").reset_index(drop=True)
    return efficient


def random_portfolios(
    expected_returns: pd.Series,
    cov_matrix: pd.DataFrame,
    n_portfolios: int = 5000,
    risk_free_rate: float = 0.02,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Simulate random long-only portfolios (weights drawn from a Dirichlet
    distribution so they sum to 1) for a background "cloud" behind the
    efficient frontier plot, illustrating that the frontier bounds the
    achievable risk/return region.
    """
    rng = np.random.default_rng(seed)
    n_assets = len(expected_returns)
    weights_matrix = rng.dirichlet(np.ones(n_assets), size=n_portfolios)

    records = []
    for w in weights_matrix:
        ret, vol = portfolio_performance(w, expected_returns, cov_matrix)
        sharpe = (ret - risk_free_rate) / vol if vol > 0 else np.nan
        records.append({"return": ret, "volatility": vol, "sharpe_ratio": sharpe})
    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def plot_covariance_heatmap(cov_matrix: pd.DataFrame, figsize=(6, 5)):
    """Heatmap of the annualized covariance matrix across tickers."""
    import matplotlib.pyplot as plt
    import seaborn as sns

    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(cov_matrix, annot=True, fmt=".4f", cmap="viridis", ax=ax)
    ax.set_title("Annualized Covariance Matrix — Daily Returns")
    fig.tight_layout()
    return fig


def plot_efficient_frontier(
    frontier: pd.DataFrame,
    max_sharpe: dict,
    min_vol: dict,
    random_cloud: pd.DataFrame = None,
    figsize=(10, 7),
):
    """
    Plot the efficient frontier curve, with an optional random-portfolio
    cloud in the background, and the Max Sharpe and Min Volatility
    portfolios marked distinctly.
    """
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=figsize)

    if random_cloud is not None:
        scatter = ax.scatter(
            random_cloud["volatility"], random_cloud["return"],
            c=random_cloud["sharpe_ratio"], cmap="viridis", s=8, alpha=0.4,
        )
        fig.colorbar(scatter, ax=ax, label="Sharpe Ratio")

    ax.plot(frontier["volatility"], frontier["target_return"], color="black", linewidth=2, label="Efficient Frontier")

    ax.scatter(
        max_sharpe["volatility"], max_sharpe["expected_return"],
        marker="*", color="red", s=400, edgecolors="black", label="Max Sharpe Ratio Portfolio", zorder=5,
    )
    ax.scatter(
        min_vol["volatility"], min_vol["expected_return"],
        marker="*", color="gold", s=400, edgecolors="black", label="Min Volatility Portfolio", zorder=5,
    )

    ax.set_xlabel("Volatility (Annualized Std. Dev.)")
    ax.set_ylabel("Expected Return (Annualized)")
    ax.set_title("Efficient Frontier — TSLA / BND / SPY")
    ax.legend()
    fig.tight_layout()
    return fig
