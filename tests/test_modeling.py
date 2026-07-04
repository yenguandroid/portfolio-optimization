import numpy as np
import pandas as pd
import pytest

from src.modeling import chronological_split, create_sequences


def _make_df(n=100, start="2024-01-01"):
    dates = pd.date_range(start, periods=n, freq="D")
    return pd.DataFrame({"Date": dates, "Close": np.linspace(100, 200, n)})


def test_chronological_split_preserves_order():
    df = _make_df(n=100)
    train, test = chronological_split(df, split_date="2024-03-01")
    assert train["Date"].max() < pd.Timestamp("2024-03-01")
    assert test["Date"].min() >= pd.Timestamp("2024-03-01")
    assert len(train) + len(test) == len(df)


def test_chronological_split_no_shuffling_train_is_earliest():
    df = _make_df(n=50)
    train, test = chronological_split(df, split_date="2024-01-25")
    # train should be exactly the first N rows in original order
    assert train["Date"].is_monotonic_increasing
    assert test["Date"].is_monotonic_increasing
    assert train["Date"].iloc[-1] < test["Date"].iloc[0]


def test_chronological_split_raises_on_empty_train_or_test():
    df = _make_df(n=10, start="2024-01-01")
    with pytest.raises(ValueError):
        chronological_split(df, split_date="2020-01-01")  # before all data -> empty train
    with pytest.raises(ValueError):
        chronological_split(df, split_date="2030-01-01")  # after all data -> empty test


def test_create_sequences_shapes():
    values = np.arange(0, 110)  # 110 points
    window = 60
    X, y = create_sequences(values, window=window)
    assert X.shape == (110 - window, window, 1)
    assert y.shape == (110 - window,)


def test_create_sequences_values_correct():
    values = np.array([1, 2, 3, 4, 5, 6, 7], dtype=float)
    X, y = create_sequences(values, window=3)
    # first sequence: [1,2,3] -> predicts 4
    assert np.array_equal(X[0].flatten(), [1, 2, 3])
    assert y[0] == 4
    # last sequence: [4,5,6] -> predicts 7
    assert np.array_equal(X[-1].flatten(), [4, 5, 6])
    assert y[-1] == 7


def test_create_sequences_raises_if_window_too_large():
    values = np.arange(5)
    with pytest.raises(ValueError):
        create_sequences(values, window=10)
