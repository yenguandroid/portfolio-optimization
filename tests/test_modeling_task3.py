import numpy as np
import pandas as pd
import pytest

from src import modeling


def _stub_arima(monkeypatch):
    """
    Replace fit_final_arima / forecast_final_arima with a deterministic,
    dependency-free fake so walk-forward indexing and future-date logic
    can be tested without requiring statsmodels to actually fit anything.
    The real fit/forecast functions are covered separately in
    test_fit_final_arima_and_forecast_final_arima_roundtrip.
    """
    def fake_fit(train_series, order, seasonal_order=(0, 0, 0, 0)):
        return {"last_value": train_series.iloc[-1]}

    def fake_forecast(results, n_periods, alpha=0.05):
        val = results["last_value"]
        forecast = np.full(n_periods, val, dtype=float)
        conf_int = np.column_stack([forecast - 1, forecast + 1])
        return forecast, conf_int

    monkeypatch.setattr(modeling, "fit_final_arima", fake_fit)
    monkeypatch.setattr(modeling, "forecast_final_arima", fake_forecast)


def _make_series(n=200, start="2024-01-01"):
    dates = pd.date_range(start, periods=n, freq="B")
    values = np.linspace(100, 300, n)
    return pd.Series(values, index=dates)


def test_walk_forward_produces_one_row_per_fold_per_horizon(monkeypatch):
    _stub_arima(monkeypatch)
    series = _make_series(n=200)

    wf = modeling.walk_forward_arima_forecast(
        series, order=(1, 1, 1), initial_train_size=150, horizon=5, step=21
    )

    assert set(wf["step_ahead"].unique()) == {1, 2, 3, 4, 5}
    # number of folds = number of valid origins given step/horizon
    n_folds = wf["origin_date"].nunique()
    assert len(wf) == n_folds * 5


def test_walk_forward_forecast_dates_are_after_origin(monkeypatch):
    _stub_arima(monkeypatch)
    series = _make_series(n=200)

    wf = modeling.walk_forward_arima_forecast(
        series, order=(1, 1, 1), initial_train_size=150, horizon=3, step=30
    )
    assert (wf["forecast_date"] > wf["origin_date"]).all()


def test_summarize_walk_forward_shows_error_growth_by_horizon(monkeypatch):
    _stub_arima(monkeypatch)
    series = _make_series(n=200)

    wf = modeling.walk_forward_arima_forecast(
        series, order=(1, 1, 1), initial_train_size=150, horizon=5, step=21
    )
    summary = modeling.summarize_walk_forward(wf)

    assert list(summary.index) == [1, 2, 3, 4, 5]
    assert set(summary.columns) == {"n_folds", "MAE", "RMSE", "MAPE"}
    # naive "repeat last value" forecast against a rising series should show
    # monotonically increasing error the further ahead the horizon
    assert summary["MAE"].is_monotonic_increasing


def test_forecast_future_returns_expected_shape_and_future_dates(monkeypatch):
    _stub_arima(monkeypatch)
    series = _make_series(n=100)

    future = modeling.forecast_future(series, order=(1, 1, 1), n_periods=15)

    assert len(future) == 15
    assert list(future.columns) == ["forecast", "lower", "upper"]
    assert (future.index > series.index[-1]).all()
    assert (future["lower"] <= future["upper"]).all()
