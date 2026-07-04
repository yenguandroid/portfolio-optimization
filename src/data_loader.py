"""
data_loader.py

Utilities for extracting, cleaning, and persisting historical financial
data for the portfolio optimization project (TSLA, BND, SPY).
"""

from __future__ import annotations

import os
from typing import Iterable

import numpy as np
import pandas as pd
import yfinance as yf

DEFAULT_TICKERS = ["TSLA", "BND", "SPY"]
DEFAULT_START = "2015-01-01"
DEFAULT_END = "2026-06-30"


def fetch_price_data(
    tickers: Iterable[str] = DEFAULT_TICKERS,
    start: str = DEFAULT_START,
    end: str = DEFAULT_END,
    auto_adjust: bool = True,
) -> dict[str, pd.DataFrame]:
    """
    Download historical OHLCV data for each ticker via YFinance.

    Parameters
    ----------
    tickers : iterable of str
        Ticker symbols to download, e.g. ["TSLA", "BND", "SPY"].
    start, end : str
        Date range in "YYYY-MM-DD" format.
    auto_adjust : bool
        If True, prices are adjusted for splits/dividends (Close becomes
        the adjusted close, which is what we want for return calculations).

    Returns
    -------
    dict[str, pd.DataFrame]
        Mapping of ticker -> DataFrame with columns
        [Open, High, Low, Close, Volume] indexed by Date.
    """
    data = {}
    for ticker in tickers:
        df = yf.download(
            ticker,
            start=start,
            end=end,
            auto_adjust=auto_adjust,
            progress=False,
        )
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df.index.name = "Date"
        df["Ticker"] = ticker
        data[ticker] = df
    return data


def combine_price_data(data: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Stack per-ticker DataFrames into one long-format DataFrame."""
    combined = pd.concat(data.values(), axis=0)
    combined = combined.reset_index().sort_values(["Ticker", "Date"])
    return combined.reset_index(drop=True)


def clean_price_data(df: pd.DataFrame, method: str = "ffill") -> pd.DataFrame:
    """
    Clean a single-ticker or long-format price DataFrame.

    - Ensures Date is datetime and set as index (for single-ticker frames).
    - Casts numeric columns to float.
    - Handles missing values via forward-fill (default), interpolation,
      or row removal.

    Parameters
    ----------
    df : pd.DataFrame
        Raw price data (must contain a 'Close' column and, if long-format,
        a 'Ticker' column).
    method : {"ffill", "interpolate", "drop"}
        Strategy for handling missing values.
    """
    df = df.copy()

    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"])

    numeric_cols = [c for c in ["Open", "High", "Low", "Close", "Volume"] if c in df.columns]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    group_key = "Ticker" if "Ticker" in df.columns else None

    if method == "ffill":
        if group_key:
            df[numeric_cols] = df.groupby(group_key)[numeric_cols].ffill().bfill()
        else:
            df[numeric_cols] = df[numeric_cols].ffill().bfill()
    elif method == "interpolate":
        if group_key:
            df[numeric_cols] = df.groupby(group_key)[numeric_cols].apply(
                lambda g: g.interpolate(method="linear").bfill().ffill()
            ).reset_index(level=0, drop=True)
        else:
            df[numeric_cols] = df[numeric_cols].interpolate(method="linear").bfill().ffill()
    elif method == "drop":
        df = df.dropna(subset=numeric_cols)
    else:
        raise ValueError(f"Unknown method: {method}")

    return df


def add_returns(df: pd.DataFrame, price_col: str = "Close") -> pd.DataFrame:
    """Add daily simple and log returns, grouped by Ticker if present."""
    df = df.copy()
    if "Ticker" in df.columns:
        df["Daily_Return"] = df.groupby("Ticker")[price_col].pct_change()
        df["Log_Return"] = df.groupby("Ticker")[price_col].transform(
            lambda s: np.log(s / s.shift(1))
        )
    else:
        df["Daily_Return"] = df[price_col].pct_change()
        df["Log_Return"] = np.log(df[price_col] / df[price_col].shift(1))
    return df


def save_processed(df: pd.DataFrame, path: str = "data/processed/combined_prices.csv") -> str:
    """Persist the cleaned, combined DataFrame to disk."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)
    return path


def load_and_prepare(
    tickers: Iterable[str] = DEFAULT_TICKERS,
    start: str = DEFAULT_START,
    end: str = DEFAULT_END,
    fillna_method: str = "ffill",
) -> pd.DataFrame:
    """
    End-to-end convenience function: fetch, combine, clean, add returns.
    Returns a single long-format DataFrame ready for EDA.
    """
    raw = fetch_price_data(tickers, start, end)
    combined = combine_price_data(raw)
    cleaned = clean_price_data(combined, method=fillna_method)
    with_returns = add_returns(cleaned)
    return with_returns
