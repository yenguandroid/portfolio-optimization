import numpy as np
import pandas as pd

from src.evaluation import (
    mean_absolute_error,
    root_mean_squared_error,
    mean_absolute_percentage_error,
    evaluate_forecast,
    compare_models,
)


def test_mae_zero_for_perfect_prediction():
    y = np.array([1.0, 2.0, 3.0])
    assert mean_absolute_error(y, y) == 0.0


def test_mae_known_value():
    y_true = np.array([10.0, 20.0, 30.0])
    y_pred = np.array([12.0, 18.0, 33.0])
    # |2| + |2| + |3| = 7 -> mean = 7/3
    assert np.isclose(mean_absolute_error(y_true, y_pred), 7 / 3)


def test_rmse_known_value():
    y_true = np.array([0.0, 0.0])
    y_pred = np.array([3.0, 4.0])
    # sqrt((9+16)/2) = sqrt(12.5)
    assert np.isclose(root_mean_squared_error(y_true, y_pred), np.sqrt(12.5))


def test_rmse_penalizes_large_errors_more_than_mae():
    y_true = np.array([0.0, 0.0, 0.0, 0.0])
    y_pred = np.array([1.0, 1.0, 1.0, 10.0])  # one big outlier
    mae = mean_absolute_error(y_true, y_pred)
    rmse = root_mean_squared_error(y_true, y_pred)
    assert rmse > mae


def test_mape_known_value():
    y_true = np.array([100.0, 200.0])
    y_pred = np.array([110.0, 180.0])
    # |10/100| + |20/200| = 0.1 + 0.1 -> mean 0.1 -> 10%
    assert np.isclose(mean_absolute_percentage_error(y_true, y_pred), 10.0)


def test_mape_ignores_zero_true_values():
    y_true = np.array([0.0, 100.0])
    y_pred = np.array([5.0, 90.0])
    # only the second row counts: |10/100| = 0.1 -> 10%
    assert np.isclose(mean_absolute_percentage_error(y_true, y_pred), 10.0)


def test_evaluate_forecast_returns_all_three_metrics():
    y_true = np.array([100.0, 105.0, 110.0])
    y_pred = np.array([98.0, 107.0, 111.0])
    result = evaluate_forecast(y_true, y_pred)
    assert set(result.keys()) == {"MAE", "RMSE", "MAPE"}
    assert all(v >= 0 for v in result.values())


def test_compare_models_sorts_by_rmse():
    results = {
        "Bad_Model": {"MAE": 10, "RMSE": 15, "MAPE": 12},
        "Good_Model": {"MAE": 2, "RMSE": 3, "MAPE": 2.5},
    }
    table = compare_models(results)
    assert table.index[0] == "Good_Model"
    assert list(table.columns) == ["MAE", "RMSE", "MAPE"]
