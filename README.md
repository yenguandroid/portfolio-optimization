# Portfolio Optimization

Time-series forecasting and portfolio optimization for TSLA (Tesla), BND
(Vanguard Total Bond Market ETF), and SPY (S&P 500 ETF), covering
2015-01-01 through 2026-06-30.

## Project Structure

```
portfolio-optimization/
├── .vscode/                # Editor settings (pytest integration, linting)
├── .github/workflows/      # CI: runs unit tests on every push/PR
├── data/processed/         # Cleaned, combined dataset (generated, gitignored)
├── notebooks/               # Task notebooks (EDA, modeling, optimization)
├── src/                    # Reusable, unit-tested source modules
│   ├── data_loader.py       # Fetch/clean/combine price data (YFinance)
│   ├── eda.py                # Rolling stats, outlier detection, ADF test, plots
│   └── risk_metrics.py       # VaR (historical & parametric), Sharpe Ratio
├── tests/                   # pytest unit tests for src/
├── scripts/                  # CLI entry points (e.g. run_eda.py)
├── requirements.txt
└── README.md
```

## Setup

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Task 1 — Preprocess and Explore the Data

**Notebook:** [`notebooks/task1_eda.ipynb`](notebooks/task1_eda.ipynb)

Covers:
1. **Data extraction** — TSLA, BND, SPY daily OHLCV from YFinance (2015-01-01 to 2026-06-30).
2. **Cleaning** — missing-value handling (forward-fill), dtype normalization, return calculation.
3. **EDA** — closing-price trends, daily-return volatility, rolling mean/std, outlier detection,
   return-distribution and correlation analysis.
4. **Stationarity** — Augmented Dickey-Fuller test on price levels vs. daily returns, with
   implications for ARIMA differencing (`d` parameter).
5. **Risk metrics** — historical & parametric Value at Risk (95%), annualized Sharpe Ratio.

Run it directly:

```bash
jupyter notebook notebooks/task1_eda.ipynb
```

Or run the equivalent pipeline from the command line (fetches data and prints a risk summary):

```bash
python scripts/run_eda.py
```

## Running Tests

```bash
pytest tests/ --cov=src -v
```

Tests use synthetic data and do not require network access, so they run cleanly in CI
(see `.github/workflows/unittests.yml`).

## Notes

- YFinance requires an internet connection at runtime; the notebook/script cannot fetch live
  data in network-restricted environments.
- `data/processed/` is where the cleaned, combined dataset is saved as CSV; raw pulls and large
  processed files are gitignored by default (see `.gitignore`) to keep the repository small.
