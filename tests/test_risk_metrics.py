import numpy as np
import pandas as pd

from src.risk_metrics import historical_var, parametric_var, sharpe_ratio, summarize_risk


def _make_returns(n=1000, mean=0.001, std=0.02, seed=42):
    rng = np.random.default_rng(seed)
    return pd.Series(rng.normal(mean, std, n))


def test_historical_var_is_positive_for_typical_returns():
    r = _make_returns()
    var = historical_var(r, confidence=0.95)
    assert var > 0


def test_parametric_var_close_to_historical_for_normal_data():
    r = _make_returns(n=20000)
    hist = historical_var(r, confidence=0.95)
    param = parametric_var(r, confidence=0.95)
    assert abs(hist - param) < 0.01


def test_sharpe_ratio_positive_for_positive_mean_returns():
    r = _make_returns(mean=0.002, std=0.01)
    sr = sharpe_ratio(r, risk_free_rate_annual=0.0)
    assert sr > 0


def test_sharpe_ratio_nan_for_zero_variance():
    r = pd.Series([0.001] * 100)
    sr = sharpe_ratio(r)
    assert np.isnan(sr)


def test_summarize_risk_returns_expected_columns():
    df = pd.DataFrame({
        "Ticker": ["TSLA"] * 500 + ["SPY"] * 500,
        "Daily_Return": np.concatenate([
            _make_returns(500, mean=0.002, std=0.03).values,
            _make_returns(500, mean=0.0005, std=0.01).values,
        ]),
    })
    summary = summarize_risk(df)
    assert set(["TSLA", "SPY"]) == set(summary.index)
    assert "Sharpe_Ratio" in summary.columns
    assert "Annualized_Volatility" in summary.columns
