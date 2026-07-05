"""
modeling.py

Model-building utilities for Task 2 (chronological train/test splitting,
ARIMA/SARIMA fitting via pmdarima's auto_arima, LSTM sequence preparation
+ architecture definition) and Task 3 (walk-forward/rolling-origin
short-horizon validation, and final future forecasting with confidence
intervals).
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Chronological train/test split
# ---------------------------------------------------------------------------

def chronological_split(
    df: pd.DataFrame,
    split_date: str,
    date_col: str = "Date",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split a single-ticker, date-sorted DataFrame into train/test sets by
    date, preserving temporal order. No shuffling — rows before
    `split_date` go to train, rows on/after go to test.

    Parameters
    ----------
    df : pd.DataFrame
        Must be sorted by date and contain `date_col`.
    split_date : str
        e.g. "2025-01-01". Train = [start, split_date). Test = [split_date, end].
    """
    df = df.sort_values(date_col).reset_index(drop=True)
    split_ts = pd.Timestamp(split_date)
    train = df[df[date_col] < split_ts].reset_index(drop=True)
    test = df[df[date_col] >= split_ts].reset_index(drop=True)
    if len(train) == 0 or len(test) == 0:
        raise ValueError(
            f"Split produced an empty set (train={len(train)}, test={len(test)}). "
            "Check that split_date falls strictly inside the data's date range."
        )
    return train, test


# ---------------------------------------------------------------------------
# ARIMA / SARIMA
#
# Reviewer feedback (Task 2 interim review) asked for a clearer separation
# between (a) MODEL SELECTION — searching the (p,d,q)/(P,D,Q,m) space to find
# the best order — and (b) FINAL REFIT — actually fitting that chosen order
# for forecasting. The two are kept as distinct functions below so the
# notebook can show "here's the order the search chose" and "here's the
# model actually used to forecast" as two separate, auditable steps rather
# than one opaque call that does both at once.
# ---------------------------------------------------------------------------

def select_arima_order(
    train_series: pd.Series,
    seasonal: bool = False,
    m: int = 1,
    **kwargs,
) -> dict:
    """
    MODEL SELECTION STEP ONLY.

    Runs pmdarima's auto_arima purely to search the (p, d, q) — and,
    if seasonal=True, (P, D, Q, m) — space and identify the
    information-criterion-minimizing order. Does NOT return a model
    meant for forecasting; it returns the chosen order as plain data,
    so the selection decision is documented independently of whatever
    object ends up doing the forecasting.

    Parameters
    ----------
    train_series : pd.Series
        Training target (typically Close price).
    seasonal : bool
        Whether to search seasonal (P, D, Q, m) terms too.
    m : int
        Seasonal period (e.g. 5 for a business-day "weekly" cycle).
        Only used if seasonal=True.
    kwargs :
        Passed through to pmdarima.auto_arima (e.g. max_p, max_q,
        stepwise, trace).

    Returns
    -------
    dict with keys: order, seasonal_order, aic, seasonal, m
    """
    import pmdarima as pm

    search_model = pm.auto_arima(
        train_series,
        seasonal=seasonal,
        m=m if seasonal else 1,
        trace=kwargs.pop("trace", False),
        error_action="ignore",
        suppress_warnings=True,
        stepwise=kwargs.pop("stepwise", True),
        **kwargs,
    )
    seasonal_order = search_model.seasonal_order if seasonal else (0, 0, 0, 0)
    return {
        "order": search_model.order,
        "seasonal_order": seasonal_order,
        "aic": search_model.aic(),
        "seasonal": seasonal,
        "m": m,
    }


def fit_final_arima(
    train_series: pd.Series,
    order: tuple,
    seasonal_order: tuple = (0, 0, 0, 0),
):
    """
    FINAL REFIT STEP.

    Explicitly fits a statsmodels SARIMAX model using the order chosen
    during `select_arima_order`. Using statsmodels directly (rather than
    re-using the pmdarima search object) makes the refit an intentional,
    separate action, and gives access to `get_forecast()` with proper
    confidence intervals for the test period.

    Parameters
    ----------
    train_series : pd.Series
        Training target the final model is fit on.
    order : tuple
        (p, d, q) chosen during selection.
    seasonal_order : tuple
        (P, D, Q, m) chosen during selection; defaults to no seasonal
        component.

    Returns
    -------
    A fitted statsmodels SARIMAXResults object.
    """
    from statsmodels.tsa.statespace.sarimax import SARIMAX

    final_model = SARIMAX(
        train_series,
        order=order,
        seasonal_order=seasonal_order,
        enforce_stationarity=False,
        enforce_invertibility=False,
    )
    final_results = final_model.fit(disp=False)
    return final_results


def forecast_final_arima(final_results, n_periods: int, alpha: float = 0.05):
    """
    Generate an n_periods-ahead forecast (with confidence interval) from
    the FINAL REFIT model produced by `fit_final_arima`.

    Returns
    -------
    forecast_mean : np.ndarray, shape (n_periods,)
    conf_int : np.ndarray, shape (n_periods, 2)  — [lower, upper] at
        confidence level (1 - alpha).
    """
    forecast_obj = final_results.get_forecast(steps=n_periods)
    forecast_mean = np.asarray(forecast_obj.predicted_mean)
    conf_int = np.asarray(forecast_obj.conf_int(alpha=alpha))
    return forecast_mean, conf_int


def fit_auto_arima(
    train_series: pd.Series,
    seasonal: bool = False,
    m: int = 1,
    **kwargs,
):
    """
    Convenience one-shot wrapper (selection + a pmdarima-native fitted
    model in a single call). Kept for quick exploratory use — for the
    documented selection/refit workflow, prefer
    `select_arima_order` + `fit_final_arima` instead.
    """
    import pmdarima as pm

    model = pm.auto_arima(
        train_series,
        seasonal=seasonal,
        m=m if seasonal else 1,
        trace=kwargs.pop("trace", False),
        error_action="ignore",
        suppress_warnings=True,
        stepwise=kwargs.pop("stepwise", True),
        **kwargs,
    )
    return model


def forecast_arima(model, n_periods: int, return_conf_int: bool = True):
    """Generate an n_periods-ahead forecast from a fitted pmdarima model
    (i.e. the one-shot `fit_auto_arima` object, not the final-refit
    statsmodels object — use `forecast_final_arima` for that one)."""
    if return_conf_int:
        forecast, conf_int = model.predict(n_periods=n_periods, return_conf_int=True)
        return np.asarray(forecast), np.asarray(conf_int)
    forecast = model.predict(n_periods=n_periods)
    return np.asarray(forecast), None


# ---------------------------------------------------------------------------
# LSTM sequence preparation
# ---------------------------------------------------------------------------

def create_sequences(values: np.ndarray, window: int = 60) -> Tuple[np.ndarray, np.ndarray]:
    """
    Turn a 1D array of scaled values into (X, y) supervised-learning
    sequences: X[i] = values[i : i+window], y[i] = values[i+window].

    Returns
    -------
    X : np.ndarray, shape (n_samples, window, 1)
    y : np.ndarray, shape (n_samples,)
    """
    values = np.asarray(values).reshape(-1)
    if len(values) <= window:
        raise ValueError(
            f"Series length ({len(values)}) must exceed window size ({window}) "
            "to create at least one sequence."
        )
    X, y = [], []
    for i in range(len(values) - window):
        X.append(values[i:i + window])
        y.append(values[i + window])
    X = np.array(X).reshape(-1, window, 1)
    y = np.array(y)
    return X, y


def build_lstm_model(
    window: int = 60,
    lstm_units: tuple = (50, 50),
    dropout: float = 0.2,
    learning_rate: float = 0.001,
):
    """
    Build a stacked-LSTM regression model:
        Input(window, 1) -> LSTM(units[0], return_sequences=True) -> Dropout
                          -> LSTM(units[1]) -> Dropout
                          -> Dense(1)

    Compiled with Adam optimizer and MSE loss.
    """
    from tensorflow import keras
    from tensorflow.keras import layers

    model = keras.Sequential()
    model.add(layers.Input(shape=(window, 1)))
    for i, units in enumerate(lstm_units):
        return_sequences = i < len(lstm_units) - 1
        model.add(layers.LSTM(units, return_sequences=return_sequences))
        model.add(layers.Dropout(dropout))
    model.add(layers.Dense(1))

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss="mse",
        metrics=["mae"],
    )
    return model


def iterative_lstm_forecast(model, last_window: np.ndarray, n_periods: int) -> np.ndarray:
    """
    Generate a multi-step-ahead forecast by iteratively feeding the
    model's own prediction back in as the newest point in the window
    (since we don't have true future values during forecasting).

    Parameters
    ----------
    model : trained keras model
    last_window : np.ndarray, shape (window,)
        The most recent `window` scaled values from the training/test
        boundary, used as the seed sequence.
    n_periods : int
        Number of future steps to forecast.

    Returns
    -------
    np.ndarray, shape (n_periods,) of scaled predictions (inverse-transform
    with the same scaler used to prepare the training data).
    """
    window = len(last_window)
    current_seq = np.asarray(last_window, dtype=float).reshape(1, window, 1)
    preds = []
    for _ in range(n_periods):
        next_val = model.predict(current_seq, verbose=0)[0, 0]
        preds.append(next_val)
        current_seq = np.append(current_seq[:, 1:, :], [[[next_val]]], axis=1)
    return np.array(preds)


# ---------------------------------------------------------------------------
# Task 3 — Walk-forward (rolling-origin) short-horizon validation
#
# Task 2's single static forecast over an ~18-month test period showed all
# models flattening to a near-constant trajectory — an expected property of
# long-horizon iterative forecasting, not a modeling defect, but not a
# useful way to judge how good these models actually are at a horizon GMF
# would realistically act on (days to a couple of weeks). Walk-forward
# validation instead repeatedly forecasts a short horizon, scores it, then
# advances the origin and repeats, giving a far more decision-relevant
# accuracy estimate.
# ---------------------------------------------------------------------------

def walk_forward_arima_forecast(
    full_series: pd.Series,
    order: tuple,
    seasonal_order: tuple = (0, 0, 0, 0),
    initial_train_size: int = None,
    horizon: int = 5,
    step: int = 21,
) -> pd.DataFrame:
    """
    Rolling-origin walk-forward validation for a fixed ARIMA/SARIMA order.

    Starting at `initial_train_size`, repeatedly:
      1. Fit on all data available up to the current origin.
      2. Forecast `horizon` steps ahead (with confidence intervals).
      3. Record predictions against the actual values that followed.
      4. Advance the origin by `step` observations and repeat, until
         fewer than `horizon` observations remain.

    This uses a fixed, already-known-good order (found once during Task 2's
    selection step) rather than re-running order search at every origin,
    which would be prohibitively slow and isn't necessary — the goal here
    is to validate short-horizon *forecast accuracy*, not to re-litigate
    model selection at every step.

    Parameters
    ----------
    full_series : pd.Series
        Indexed by date, covering the full period to walk forward through
        (typically train + test combined).
    order, seasonal_order : tuple
        The ARIMA/SARIMA order chosen during Task 2 selection.
    initial_train_size : int
        Number of observations in the first training window (e.g. the
        length of the original Task 2 training set). Defaults to 80% of
        the series if not provided.
    horizon : int
        Number of steps to forecast at each origin (e.g. 5 = one trading
        week).
    step : int
        Number of observations to advance the origin between folds (e.g.
        21 = roughly one trading month). A smaller step gives more folds
        (more robust metrics) at the cost of more model fits.

    Returns
    -------
    pd.DataFrame with columns:
        origin_date, step_ahead, forecast_date, actual, forecast, lower, upper
    One row per (origin, step-ahead) forecast across all folds.
    """
    values = full_series.values
    dates = full_series.index
    n = len(values)

    if initial_train_size is None:
        initial_train_size = int(n * 0.8)

    records = []
    origin = initial_train_size
    while origin + horizon <= n:
        train_slice = full_series.iloc[:origin]
        results = fit_final_arima(train_slice, order=order, seasonal_order=seasonal_order)
        forecast, conf_int = forecast_final_arima(results, n_periods=horizon)

        for h in range(horizon):
            records.append({
                "origin_date": dates[origin - 1],
                "step_ahead": h + 1,
                "forecast_date": dates[origin + h],
                "actual": values[origin + h],
                "forecast": forecast[h],
                "lower": conf_int[h, 0],
                "upper": conf_int[h, 1],
            })
        origin += step

    return pd.DataFrame(records)


def summarize_walk_forward(walk_forward_df: pd.DataFrame) -> pd.DataFrame:
    """
    Summarize walk-forward results by step-ahead horizon: MAE/RMSE/MAPE at
    each forecast horizon (1-day-ahead, 2-day-ahead, ...), so accuracy
    degradation as the horizon lengthens is visible directly, rather than
    averaged away across all horizons at once.
    """
    from src.evaluation import evaluate_forecast

    rows = []
    for h, group in walk_forward_df.groupby("step_ahead"):
        metrics = evaluate_forecast(group["actual"], group["forecast"])
        metrics["step_ahead"] = h
        metrics["n_folds"] = len(group)
        rows.append(metrics)
    return pd.DataFrame(rows).set_index("step_ahead")[["n_folds", "MAE", "RMSE", "MAPE"]]


# ---------------------------------------------------------------------------
# Task 3 — Final future forecast with confidence intervals
# ---------------------------------------------------------------------------

def forecast_future(
    full_series: pd.Series,
    order: tuple,
    seasonal_order: tuple = (0, 0, 0, 0),
    n_periods: int = 180,
    alpha: float = 0.05,
    freq: str = "B",
):
    """
    Refit the chosen order on ALL available history (train + test) and
    forecast genuinely forward, beyond the last known date, with
    confidence intervals.

    This is distinct from Task 2's test-period forecast (which validated
    the model against known, held-out actuals): here there is no ground
    truth to compare against, since these dates are in the future. The
    output is a forward-looking projection intended to inform Task 4's
    portfolio construction, not a re-run of Task 2's evaluation.

    Parameters
    ----------
    full_series : pd.Series
        All available historical data (train + test combined), indexed
        by date.
    order, seasonal_order : tuple
        The chosen ARIMA/SARIMA order (carry over the winning order from
        Task 2's selection/evaluation).
    n_periods : int
        Number of future business days to forecast (e.g. 180 \u2248 ~8.5
        months of trading days, 252 \u2248 1 year).
    alpha : float
        Significance level for the confidence interval (0.05 = 95% CI).
    freq : str
        Frequency for generating future dates ("B" = business day).

    Returns
    -------
    pd.DataFrame indexed by future date, with columns
        [forecast, lower, upper]
    """
    results = fit_final_arima(full_series, order=order, seasonal_order=seasonal_order)
    forecast, conf_int = forecast_final_arima(results, n_periods=n_periods, alpha=alpha)

    last_date = full_series.index[-1]
    future_dates = pd.bdate_range(start=last_date, periods=n_periods + 1, freq=freq)[1:]

    return pd.DataFrame(
        {
            "forecast": forecast,
            "lower": conf_int[:, 0],
            "upper": conf_int[:, 1],
        },
        index=future_dates,
    )
