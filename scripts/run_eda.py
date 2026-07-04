"""
run_eda.py

Command-line entry point that fetches, cleans, and persists the
TSLA / BND / SPY dataset used throughout the project. Run this once
to populate data/processed/combined_prices.csv before opening the
notebook, or let the notebook call load_and_prepare() directly.

Usage:
    python scripts/run_eda.py
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.data_loader import load_and_prepare, save_processed  # noqa: E402
from src.risk_metrics import summarize_risk  # noqa: E402


def main():
    print("Fetching TSLA, BND, SPY from YFinance (2015-01-01 to 2026-06-30)...")
    df = load_and_prepare()
    path = save_processed(df)
    print(f"Saved cleaned, combined dataset to: {path}")

    print("\nRisk summary:")
    summary = summarize_risk(df)
    print(summary)


if __name__ == "__main__":
    main()
