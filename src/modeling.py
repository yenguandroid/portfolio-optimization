"""
modeling.py

Model-building utilities for Task 2: chronological train/test splitting,
ARIMA/SARIMA fitting (via pmdarima's auto_arima), and LSTM sequence
preparation + architecture definition.
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
# ---------------------------------------------------------------------------

def fit_auto_arima(
    train_series: pd.Series,
    seasonal: bool = False,
    m: int = 1,
    **kwargs,
):
    """
    Fit an auto_arima model (pmdarima) on a training series, searching
    for the best (p, d, q) — and (P, D, Q, m) if seasonal=True — by
    information criterion (AIC by default).

    This is a thin wrapper so the notebook doesn't need to import
    pmdarima directly, and so the exact search config is documented
    and testable in one place.

    Parameters
    ----------
    train_series : pd.Series
        Training target (typically Close price), indexed by date or
        a simple RangeIndex.
    seasonal : bool
        Whether to search seasonal (P, D, Q, m) terms too.
    m : int
        Seasonal period (e.g. 5 for a business-day "weekly" cycle,
        252 for an annual cycle in daily trading data). Only used if
        seasonal=True.
    kwargs :
        Passed through to pmdarima.auto_arima (e.g. max_p, max_q,
        stepwise, trace).
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
    """Generate an n_periods-ahead forecast from a fitted pmdarima model."""
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
