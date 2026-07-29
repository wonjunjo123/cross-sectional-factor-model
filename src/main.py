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
from backtest import (
    compare_models, compare_portfolio_constructions,
    compute_portfolio_returns, apply_transaction_costs,
)
from visualize import plot_model_comparison, plot_ic_timeseries, plot_equity_curve

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

    print("Training XGBoost ranker on walk-forward...")
    
    # Returns a dataframe of out-of-sample predictions with columns: [date, permno, fwd_ret, pred]
    # this is what backtest.py consumes
    preds = run_walk_forward(feature_panel, test_months=1, step_months=HORIZON, horizon=HORIZON)

    print("-- IC summary --")
    ic = summarize_ic(preds)
    ic.to_csv(OUTPUT_DIR / "ic_gbm.csv", index=False)

    preds.to_parquet(OUTPUT_DIR / "predictions_gbm.parquet", index=False)

    print("\n--- Performance summary ---")
    # 12 / HORIZON (true division, not //): freq is just an annualization
    # scalar (ann_return = ret.mean() * freq), so it doesn't need to be a
    # whole number -- and // would silently floor to the wrong factor for
    # any HORIZON that doesn't evenly divide 12 (e.g. HORIZON=5 -> 12//5=2
    # instead of the correct 2.4).
    summary = compare_models({"gbm": preds}, freq=12 / HORIZON)
    print(summary)
    summary.to_csv(OUTPUT_DIR / "model_comparison.csv")
    plot_model_comparison(OUTPUT_DIR / "model_comparison.csv", OUTPUT_DIR / "model_comparison.png")
    plot_ic_timeseries({"gbm": ic}, OUTPUT_DIR / "ic_timeseries.png")

    print("\n--- Long-short vs. long-only ---")
    construction_summary = compare_portfolio_constructions(preds, freq=12 / HORIZON)
    print(construction_summary)
    construction_summary.to_csv(OUTPUT_DIR / "portfolio_construction_comparison.csv")

    # compare_models/compare_portfolio_constructions only keep the
    # annualized summary -- save the underlying per-quarter return series
    # too, since that's what an equity curve actually needs to plot.
    portfolio_returns = compute_portfolio_returns(preds)
    portfolio_returns = apply_transaction_costs(portfolio_returns, preds)
    portfolio_returns.to_csv(OUTPUT_DIR / "portfolio_returns.csv", index=False)
    plot_equity_curve(OUTPUT_DIR / "portfolio_returns.csv", OUTPUT_DIR / "equity_curve.png")

if __name__ == "__main__":
    main()
