"""
evaluation.py

Forecast evaluation metrics (MAE, RMSE, MAPE) and a helper to assemble
a side-by-side model comparison table.
"""

from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd


def mean_absolute_error(y_true, y_pred) -> float:
    y_true, y_pred = np.asarray(y_true, dtype=float), np.asarray(y_pred, dtype=float)
    return float(np.mean(np.abs(y_true - y_pred)))


def root_mean_squared_error(y_true, y_pred) -> float:
    y_true, y_pred = np.asarray(y_true, dtype=float), np.asarray(y_pred, dtype=float)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mean_absolute_percentage_error(y_true, y_pred) -> float:
    """
    MAPE as a percentage. Rows where y_true == 0 are excluded to avoid
    division by zero (stock closing prices are never exactly zero in
    practice, but this guards against degenerate inputs, e.g. in tests).
    """
    y_true, y_pred = np.asarray(y_true, dtype=float), np.asarray(y_pred, dtype=float)
    mask = y_true != 0
    if not mask.any():
        return float("nan")
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)


def evaluate_forecast(y_true, y_pred) -> Dict[str, float]:
    """Compute MAE, RMSE, and MAPE for one set of forecasts."""
    return {
        "MAE": mean_absolute_error(y_true, y_pred),
        "RMSE": root_mean_squared_error(y_true, y_pred),
        "MAPE": mean_absolute_percentage_error(y_true, y_pred),
    }


def compare_models(results: Dict[str, Dict[str, float]]) -> pd.DataFrame:
    """
    Build a comparison table from a dict of {model_name: {metric: value}}.

    Example
    -------
    >>> compare_models({
    ...     "ARIMA": evaluate_forecast(y_true, arima_preds),
    ...     "LSTM": evaluate_forecast(y_true, lstm_preds),
    ... })
    """
    table = pd.DataFrame(results).T
    table = table[["MAE", "RMSE", "MAPE"]]
    table.index.name = "Model"
    return table.sort_values("RMSE")
