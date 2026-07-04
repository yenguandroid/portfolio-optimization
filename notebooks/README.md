# Notebooks

| Notebook | Task | Description |
|---|---|---|
| `task1_eda.ipynb` | Task 1 | Data extraction, cleaning, EDA, stationarity testing, and risk metrics for TSLA, BND, and SPY (2015-01-01 – 2026-06-30). |

## Running

From the project root, with the virtual environment activated and
dependencies from `requirements.txt` installed:

```bash
jupyter notebook notebooks/task1_eda.ipynb
```

The notebook imports reusable logic from `src/` (`data_loader.py`,
`eda.py`, `risk_metrics.py`) rather than duplicating it, so any bug
fixes or improvements to those modules are picked up automatically.
