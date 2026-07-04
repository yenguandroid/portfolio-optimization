import numpy as np
import pandas as pd
import pytest

from src.data_loader import clean_price_data, add_returns, combine_price_data


def _make_raw_df(ticker="TSLA", n=10, with_gap=True):
    dates = pd.date_range("2024-01-01", periods=n, freq="D")
    prices = np.linspace(100, 110, n)
    if with_gap:
        prices[3] = np.nan
    df = pd.DataFrame({
        "Date": dates,
        "Open": prices,
        "High": prices + 1,
        "Low": prices - 1,
        "Close": prices,
        "Volume": np.random.randint(1000, 5000, n),
        "Ticker": ticker,
    })
    return df


def test_clean_price_data_ffill_removes_nans():
    df = _make_raw_df()
    cleaned = clean_price_data(df, method="ffill")
    assert cleaned["Close"].isna().sum() == 0


def test_clean_price_data_drop_removes_rows():
    df = _make_raw_df()
    cleaned = clean_price_data(df, method="drop")
    assert cleaned["Close"].isna().sum() == 0
    assert len(cleaned) == len(df) - 1


def test_clean_price_data_invalid_method_raises():
    df = _make_raw_df()
    with pytest.raises(ValueError):
        clean_price_data(df, method="bogus")


def test_add_returns_computes_pct_change():
    df = _make_raw_df(with_gap=False)
    df = add_returns(df)
    assert "Daily_Return" in df.columns
    assert "Log_Return" in df.columns
    # first row should be NaN, rest should be finite
    assert pd.isna(df["Daily_Return"].iloc[0])
    assert df["Daily_Return"].iloc[1:].notna().all()


def test_combine_price_data_stacks_tickers():
    df1 = _make_raw_df(ticker="TSLA", with_gap=False)
    df2 = _make_raw_df(ticker="SPY", with_gap=False)
    combined = combine_price_data({"TSLA": df1.set_index("Date"), "SPY": df2.set_index("Date")})
    assert set(combined["Ticker"].unique()) == {"TSLA", "SPY"}
    assert len(combined) == len(df1) + len(df2)
