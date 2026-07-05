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


def test_select_arima_order_returns_documented_order_dict():
    """
    Model-selection step: should return plain data describing the chosen
    order (not a forecasting-ready model), so selection and final refit
    stay decoupled. Uses a small, clearly seasonal-free synthetic series
    so the search itself runs fast and offline (no network needed --
    pmdarima fits locally on the array).
    """
    from src.modeling import select_arima_order

    rng = np.random.default_rng(0)
    series = pd.Series(np.cumsum(rng.normal(0, 1, 80)) + 100)  # random-walk-like

    selection = select_arima_order(series, seasonal=False, max_p=2, max_q=2, max_d=1)

    assert set(selection.keys()) == {"order", "seasonal_order", "aic", "seasonal", "m"}
    assert len(selection["order"]) == 3
    assert isinstance(selection["aic"], float)
    assert selection["seasonal"] is False


def test_fit_final_arima_and_forecast_final_arima_roundtrip():
    """
    Final-refit step: given a chosen order, fit a statsmodels SARIMAX
    directly and confirm forecasting produces the right shapes. This
    exercises the selection/refit split end-to-end using a fixed,
    already-known order (skipping the search) to keep the test fast.
    """
    from src.modeling import fit_final_arima, forecast_final_arima

    rng = np.random.default_rng(1)
    series = pd.Series(np.cumsum(rng.normal(0, 1, 100)) + 50)

    results = fit_final_arima(series, order=(1, 1, 1), seasonal_order=(0, 0, 0, 0))
    n_periods = 10
    forecast, conf_int = forecast_final_arima(results, n_periods=n_periods)

    assert forecast.shape == (n_periods,)
    assert conf_int.shape == (n_periods, 2)
    # lower bound of CI should not exceed the upper bound
    assert (conf_int[:, 0] <= conf_int[:, 1]).all()
