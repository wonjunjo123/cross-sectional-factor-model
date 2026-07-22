"""
main.py

End-to-end orchestration: data -> features -> walk-forward models ->
backtest comparison. Run this after data_prep.py has pulled CRSP prices
and point-in-time S&P 500 membership from WRDS.

    python src/data_prep.py <wrds_username>   # one-time WRDS pull, caches to data/
    python src/main.py                        # runs the full research pipeline
"""

import pandas as pd
from pathlib import Path
from tqdm import tqdm

from features import build_feature_panel
from model import run_walk_forward, summarize_ic
from backtest import compare_models, performance_summary, compute_portfolio_returns

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "output"


def main():
    print("Loading CRSP price panel and point-in-time membership...")
    daily_panel = pd.read_parquet(DATA_DIR / "prices_wrds.parquet")
    membership = pd.read_parquet(DATA_DIR / "sp500_membership.parquet")

    print("Building feature panel...")
    feature_panel = build_feature_panel(daily_panel, membership)
    print(f"Feature panel shape: {feature_panel.shape}")
    print(f"Date range: {feature_panel['date'].min()} to {feature_panel['date'].max()}")
    print(
        f"Unique PERMNOs represented (point-in-time members only): "
        f"{feature_panel['permno'].nunique()}"
    )

    OUTPUT_DIR.mkdir(exist_ok=True)
    feature_panel.to_parquet(OUTPUT_DIR / "feature_panel.parquet", index=False)

    predictions_by_model = {}

    for model_type in ["linear", "gbm"]:
        print(f"\nRunning walk-forward for model: {model_type}")
        preds = run_walk_forward(feature_panel, model_type=model_type)
        predictions_by_model[model_type] = preds

        print(f"-- {model_type} IC summary --")
        summarize_ic(preds)

        preds.to_parquet(OUTPUT_DIR / f"predictions_{model_type}.parquet", index=False)

    print("\n--- Model comparison ---")
    comparison = compare_models(predictions_by_model)
    print(comparison)
    comparison.to_csv(OUTPUT_DIR / "model_comparison.csv")


if __name__ == "__main__":
    main()
