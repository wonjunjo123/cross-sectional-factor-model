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
    compare_portfolio_constructions,
    compute_portfolio_returns, apply_transaction_costs,
)
from visualize import (
    plot_portfolio_construction_comparison, plot_ic_timeseries, plot_equity_curve,
)

# this is relative file path for data and output
DATA_DIR = Path(__file__).resolve().parent.parent / "data"
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "output"

# Forward-return horizon in months. test_months/step_months are set equal to
# HORIZON below so each walk-forward test period's realized return window is
# disjoint from the next -- required for backtest.py to validly compound
# them as a sequential return series. See model.run_walk_forward's docstring.
HORIZON = 3
TRAIN_MONTHS = 60

# The default model parameters are already declared within model.py,
# but I just wanted to put it here so that I can remember I have the option to choose from main.py
# I can pass MODEL_PARAMS in run_walk_forward method
MODEL_PARAMS = dict(
    objective="rank:pairwise", n_estimators=100, max_depth=4,
    learning_rate=0.01, min_child_weight=5, colsample_bytree=0.7,
    verbosity=0, random_state=0,
)

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
    
    # preds returns a dataframe of out-of-sample predictions with columns: [date, permno, fwd_ret, pred]
    # preds is all of the out-of-sample predictions aggregated from all the (October, January, April, July, for example, in HORIZON number of increments)
    # this is what backtest.py consumes
    (preds, diagnostic) = run_walk_forward(feature_panel, train_months=TRAIN_MONTHS, test_months=1,
                                          step_months=HORIZON, horizon=HORIZON, return_diagnostics=True, model_params=MODEL_PARAMS)
    preds.to_csv('predictions.csv')
    print("-- IC summary --")
    ic = summarize_ic(preds) # these are the out of sample ICs
    
    ic.to_csv(OUTPUT_DIR / "ic_gbm.csv", index=False)

    preds.to_parquet(OUTPUT_DIR / "predictions_gbm.parquet", index=False)

    plot_ic_timeseries({"gbm": ic}, OUTPUT_DIR / "ic_timeseries.png")

    print("\n--- Long-short vs. long-only ---")
    # 12 / HORIZON (true division, not //): freq is just an annualization
    # scalar (ann_return = ret.mean() * freq), so it doesn't need to be a
    # whole number -- and // would silently floor to the wrong factor for
    # any HORIZON that doesn't evenly divide 12 (e.g. HORIZON=5 -> 12//5=2
    # instead of the correct 2.4).
    construction_summary = compare_portfolio_constructions(preds, freq=12 / HORIZON)
    print(construction_summary)
    construction_summary.to_csv(OUTPUT_DIR / "portfolio_construction_comparison.csv")
    plot_portfolio_construction_comparison(
        OUTPUT_DIR / "portfolio_construction_comparison.csv",
        OUTPUT_DIR / "portfolio_construction_comparison.png",
    )

    # compare_portfolio_constructions only keeps the annualized summary --
    # save the underlying per-quarter return series too, since that's what
    # an equity curve actually needs to plot.
    portfolio_returns = compute_portfolio_returns(preds)
    portfolio_returns = apply_transaction_costs(portfolio_returns, preds)
    portfolio_returns.to_csv(OUTPUT_DIR / "portfolio_returns.csv", index=False)
    plot_equity_curve(OUTPUT_DIR / "portfolio_returns.csv", OUTPUT_DIR / "equity_curve.png")

if __name__ == "__main__":
    main()
