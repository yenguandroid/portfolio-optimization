"""
eda.py

Exploratory data analysis helpers: rolling statistics, outlier detection,
stationarity testing, and plotting utilities for the price/return data.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.stattools import adfuller

sns.set_theme(style="whitegrid")


# ---------------------------------------------------------------------------
# Rolling statistics
# ---------------------------------------------------------------------------

def add_rolling_stats(df: pd.DataFrame, window: int = 30, col: str = "Daily_Return") -> pd.DataFrame:
    """Add rolling mean and rolling std (annualized-friendly) per ticker."""
    df = df.copy()
    if "Ticker" in df.columns:
        df[f"Rolling_Mean_{window}"] = df.groupby("Ticker")[col].transform(
            lambda s: s.rolling(window).mean()
        )
        df[f"Rolling_Std_{window}"] = df.groupby("Ticker")[col].transform(
            lambda s: s.rolling(window).std()
        )
    else:
        df[f"Rolling_Mean_{window}"] = df[col].rolling(window).mean()
        df[f"Rolling_Std_{window}"] = df[col].rolling(window).std()
    return df


# ---------------------------------------------------------------------------
# Outlier detection
# ---------------------------------------------------------------------------

def detect_outliers_zscore(df: pd.DataFrame, col: str = "Daily_Return", threshold: float = 3.0) -> pd.DataFrame:
    """
    Flag rows whose Daily_Return z-score exceeds `threshold` (per ticker).
    Returns only the flagged rows, sorted by absolute return.
    """
    df = df.copy()

    def _zscore(s: pd.Series) -> pd.Series:
        return (s - s.mean()) / s.std()

    if "Ticker" in df.columns:
        df["Z_Score"] = df.groupby("Ticker")[col].transform(_zscore)
    else:
        df["Z_Score"] = _zscore(df[col])

    outliers = df[df["Z_Score"].abs() > threshold].copy()
    outliers["Abs_Return"] = outliers[col].abs()
    return outliers.sort_values("Abs_Return", ascending=False)


# ---------------------------------------------------------------------------
# Stationarity testing
# ---------------------------------------------------------------------------

def adf_test(series: pd.Series, name: str = "") -> dict:
    """
    Run the Augmented Dickey-Fuller test on a series (drops NaNs first).

    Returns a dict with the test statistic, p-value, critical values,
    and a plain-English verdict.
    """
    clean_series = series.dropna()
    result = adfuller(clean_series, autolag="AIC")
    output = {
        "name": name,
        "ADF_Statistic": result[0],
        "p_value": result[1],
        "n_lags": result[2],
        "n_obs": result[3],
        "critical_values": result[4],
        "is_stationary": result[1] < 0.05,
    }
    return output


def print_adf_result(result: dict) -> None:
    verdict = "STATIONARY" if result["is_stationary"] else "NON-STATIONARY"
    print(f"--- ADF Test: {result['name']} ---")
    print(f"ADF Statistic : {result['ADF_Statistic']:.4f}")
    print(f"p-value       : {result['p_value']:.4f}")
    for key, val in result["critical_values"].items():
        print(f"Critical Value ({key}) : {val:.4f}")
    print(f"Verdict       : {verdict} (alpha=0.05)\n")


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_closing_prices(df: pd.DataFrame, tickers: list[str] | None = None, figsize=(12, 6)):
    tickers = tickers or df["Ticker"].unique().tolist()
    fig, ax = plt.subplots(figsize=figsize)
    for t in tickers:
        subset = df[df["Ticker"] == t]
        ax.plot(subset["Date"], subset["Close"], label=t)
    ax.set_title("Adjusted Closing Price Over Time")
    ax.set_xlabel("Date")
    ax.set_ylabel("Price (USD)")
    ax.legend()
    fig.tight_layout()
    return fig


def plot_daily_returns(df: pd.DataFrame, ticker: str, figsize=(12, 4)):
    subset = df[df["Ticker"] == ticker]
    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(subset["Date"], subset["Daily_Return"], color="steelblue", linewidth=0.7)
    ax.set_title(f"{ticker} Daily Percentage Return")
    ax.set_xlabel("Date")
    ax.set_ylabel("Daily Return")
    fig.tight_layout()
    return fig


def plot_rolling_mean(df: pd.DataFrame, ticker: str, window: int = 30, figsize=(12, 4)):
    """
    Plot the rolling mean of daily returns for a single ticker — the
    short-term drift/momentum signal, as distinct from rolling volatility
    (see plot_rolling_volatility). A rolling mean sitting persistently
    above zero indicates a short-term upward drift in returns; below zero
    indicates downward drift; oscillation around zero indicates no
    persistent short-term trend in returns.
    """
    subset = df[df["Ticker"] == ticker]
    col = f"Rolling_Mean_{window}"
    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(subset["Date"], subset[col], color="mediumblue")
    ax.axhline(0, color="grey", linestyle="--", linewidth=1)
    ax.set_title(f"{ticker} Rolling {window}-Day Mean of Daily Return")
    ax.set_xlabel("Date")
    ax.set_ylabel("Rolling Mean Return")
    fig.tight_layout()
    return fig


def plot_rolling_mean_and_volatility(df: pd.DataFrame, ticker: str, window: int = 30, figsize=(12, 6)):
    """
    Stacked two-panel view of rolling mean (drift) and rolling std
    (volatility) of daily returns for a single ticker, sharing a date
    axis so short-term trend and short-term risk can be read side by
    side (e.g. spotting periods where volatility spikes while drift
    turns negative, a classic market-stress signature).
    """
    subset = df[df["Ticker"] == ticker]
    mean_col = f"Rolling_Mean_{window}"
    std_col = f"Rolling_Std_{window}"

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, sharex=True)
    ax1.plot(subset["Date"], subset[mean_col], color="mediumblue")
    ax1.axhline(0, color="grey", linestyle="--", linewidth=1)
    ax1.set_title(f"{ticker} Rolling {window}-Day Mean of Daily Return (Drift)")
    ax1.set_ylabel("Rolling Mean Return")

    ax2.plot(subset["Date"], subset[std_col], color="darkorange")
    ax2.set_title(f"{ticker} Rolling {window}-Day Std of Daily Return (Volatility)")
    ax2.set_xlabel("Date")
    ax2.set_ylabel("Rolling Std Dev")

    fig.tight_layout()
    return fig


def plot_rolling_volatility(df: pd.DataFrame, ticker: str, window: int = 30, figsize=(12, 4)):
    subset = df[df["Ticker"] == ticker]
    col = f"Rolling_Std_{window}"
    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(subset["Date"], subset[col], color="darkorange")
    ax.set_title(f"{ticker} Rolling {window}-Day Volatility (Std of Daily Return)")
    ax.set_xlabel("Date")
    ax.set_ylabel("Rolling Std Dev")
    fig.tight_layout()
    return fig


def plot_return_distribution(df: pd.DataFrame, ticker: str, figsize=(8, 5)):
    subset = df[df["Ticker"] == ticker]["Daily_Return"].dropna()
    fig, ax = plt.subplots(figsize=figsize)
    sns.histplot(subset, kde=True, bins=100, ax=ax, color="mediumseagreen")
    ax.set_title(f"{ticker} Distribution of Daily Returns")
    ax.set_xlabel("Daily Return")
    fig.tight_layout()
    return fig


def plot_outliers(df: pd.DataFrame, ticker: str, outliers: pd.DataFrame, figsize=(12, 5)):
    subset = df[df["Ticker"] == ticker]
    tick_outliers = outliers[outliers["Ticker"] == ticker] if "Ticker" in outliers.columns else outliers
    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(subset["Date"], subset["Daily_Return"], color="lightgrey", linewidth=0.7, label="Daily Return")
    ax.scatter(tick_outliers["Date"], tick_outliers["Daily_Return"], color="red", s=25, label="Outlier", zorder=5)
    ax.set_title(f"{ticker} Outlier Days (|Z-score| > 3)")
    ax.set_xlabel("Date")
    ax.set_ylabel("Daily Return")
    ax.legend()
    fig.tight_layout()
    return fig


def plot_correlation_heatmap(df: pd.DataFrame, figsize=(6, 5)):
    """Correlation of daily returns across tickers (requires wide format)."""
    wide = df.pivot(index="Date", columns="Ticker", values="Daily_Return")
    corr = wide.corr()
    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(corr, annot=True, cmap="coolwarm", vmin=-1, vmax=1, ax=ax)
    ax.set_title("Correlation of Daily Returns")
    fig.tight_layout()
    return fig
