"""
main.py

End-to-end orchestration: data -> features -> walk-forward model ->
backtest. Run this after data_prep.py has pulled CRSP prices and
point-in-time S&P 500 membership from WRDS.

    python src/data_prep.py <wrds_username>   # one-time WRDS pull, caches to data/
    python src/main.py                        # runs the full research pipeline
"""

import pandas as pd
from pathlib import Path

from features import build_feature_panel
from model import run_walk_forward, summarize_ic
from backtest import compare_models
from visualize import plot_model_comparison, plot_ic_timeseries

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "output"

# Forward-return horizon in months. test_months/step_months are set equal to
# HORIZON below so each walk-forward test period's realized return window is
# disjoint from the next -- required for backtest.py to validly compound
# them as a sequential return series. See model.run_walk_forward's docstring.
HORIZON = 3


def main():
    print("Loading CRSP price panel and point-in-time membership...")
    daily_panel = pd.read_parquet(DATA_DIR / "prices_wrds.parquet")
    membership = pd.read_parquet(DATA_DIR / "sp500_membership.parquet")
    
    print("Building feature panel...")
    feature_panel = build_feature_panel(daily_panel, membership, horizon=HORIZON)
    print(f"Feature panel shape: {feature_panel.shape}")
    print(f"Date range: {feature_panel['date'].min()} to {feature_panel['date'].max()}")
    print(
        f"Unique PERMNOs represented (point-in-time members only): "
        f"{feature_panel['permno'].nunique()}"
    )

    OUTPUT_DIR.mkdir(exist_ok=True)
    feature_panel.to_parquet(OUTPUT_DIR / "feature_panel.parquet", index=False)

    print("Training LightGBM ranker on walk-forward...")
    
    # Returns a dataframe of out-of-sample predictions with columns: [date, permno, fwd_ret, pred]
    # this is what backtest.py consumes
    preds = run_walk_forward(feature_panel, test_months=1, step_months=HORIZON, horizon=HORIZON)

    print("-- IC summary --")
    ic = summarize_ic(preds)
    ic.to_csv(OUTPUT_DIR / "ic_gbm.csv", index=False)

    preds.to_parquet(OUTPUT_DIR / "predictions_gbm.parquet", index=False)

    print("\n--- Performance summary ---")
    summary = compare_models({"gbm": preds}, freq=12 // HORIZON)
    print(summary)
    summary.to_csv(OUTPUT_DIR / "model_comparison.csv")
    plot_model_comparison(OUTPUT_DIR / "model_comparison.csv", OUTPUT_DIR / "model_comparison.png")
    plot_ic_timeseries({"gbm": ic}, OUTPUT_DIR / "ic_timeseries.png")

if __name__ == "__main__":
    main()
