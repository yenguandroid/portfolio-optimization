# Notebooks

| Notebook | Task | Description |
|---|---|---|
| `task1_eda.ipynb` | Task 1 | Data extraction, cleaning, EDA, stationarity testing, and risk metrics for TSLA, BND, and SPY (2015-01-01 – 2026-06-30). |
| `task2_forecasting.ipynb` | Task 2 | Chronological train/test split, ARIMA/SARIMA (via `auto_arima`) and LSTM forecasting models for TSLA closing price, with MAE/RMSE/MAPE comparison. Requires `data/processed/combined_prices.csv` from Task 1. |
| `task3_future_forecast.ipynb` | Task 3 | Walk-forward (rolling-origin) short-horizon validation, and a genuine future TSLA price forecast (beyond the dataset's end date) with widening confidence intervals. Requires Task 1's processed data and reuses the winning order identified in Task 2. |

## Running

From the project root, with the virtual environment activated and
dependencies from `requirements.txt` installed:

```bash
jupyter notebook notebooks/task1_eda.ipynb
```

The notebook imports reusable logic from `src/` (`data_loader.py`,
`eda.py`, `risk_metrics.py`) rather than duplicating it, so any bug
fixes or improvements to those modules are picked up automatically.
