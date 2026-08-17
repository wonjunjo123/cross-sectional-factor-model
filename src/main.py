"""
main.py

End-to-end orchestration: data -> features -> walk-forward model ->
backtest. Run this after api_data_prep.py (or load_wrds_web_export.py)
has pulled CRSP prices and point-in-time S&P 500 membership, and
data_prep_fundamentals.py (or load_wrds_fundamentals.py) has pulled the
IBES/short-interest/sector fundamentals added by
feature_expansion_instructions.md.

    python src/api_data_prep.py <wrds_username>  # CRSP prices + membership
    python src/data_prep_fundamentals.py          # IBES/short-interest/sector
    python src/main.py                            # runs the full research pipeline
"""

import pandas as pd
from pathlib import Path
import pickle

from features import build_feature_panel
from model import run_walk_forward, summarize_ic, summarize_tails_ic
from baseline import run_linear_walk_forward
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
    objective="rank:pairwise", n_estimators=12, max_depth=4,
    learning_rate=0.001, min_child_weight=30, colsample_bytree=0.6, subsample=0.7,
    verbosity=0, random_state=0,
)
#before was max_depth, min_child_weight, colsample_bytree, subsample = (4,5, 0.7, didn't choose)
#max_depth, min_child_weight: (2,30) -> oos_ic_mean=+0.0099  oos_ic_ir=+0.111 
#I used to do n_estimators = 100 with learning_rate 0.01
# But now I'm doing n_estimators = 50 with learning_rate 0.001

def main():
    print("Loading CRSP price panel and point-in-time membership...")
    daily_panel = pd.read_parquet(DATA_DIR / "prices_wrds.parquet")
    membership = pd.read_parquet(DATA_DIR / "sp500_membership.parquet")

    print("Loading fundamentals (IBES estimates/actuals, short interest, sector)...")
    ibes_estimates = pd.read_parquet(DATA_DIR / "ibes_estimates.parquet")
    ibes_actuals = pd.read_parquet(DATA_DIR / "ibes_actuals.parquet")
    short_interest = pd.read_parquet(DATA_DIR / "short_interest.parquet")
    sector_crosswalk = pd.read_parquet(DATA_DIR / "sector_crosswalk.parquet")
    siccd_history = pd.read_parquet(DATA_DIR / "siccd_history.parquet")

    print("Building feature panel...")
    feature_panel = build_feature_panel(
        daily_panel, membership, ibes_estimates, ibes_actuals, short_interest,
        sector_crosswalk, siccd_history, horizon=HORIZON,
    )
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
    
    print("-- IC summary --")
    ic = summarize_ic(preds) # these are the out of sample ICs
    ic.to_csv(OUTPUT_DIR / "ic_gbm.csv", index=False)

    # Tails-only counterpart: restricted to the predicted top/bottom
    # decile each month (what the long-short portfolio actually trades)
    # rather than the full cross-section -- see summarize_tails_ic's
    # docstring for why this can diverge from the full-sample IC above.
    print("\n-- IC summary (top/bottom decile only) --")
    ic_tails = summarize_tails_ic(preds)
    ic_tails.to_csv(OUTPUT_DIR / "ic_gbm_tails.csv", index=False)

    preds.to_parquet(OUTPUT_DIR / "predictions_gbm.parquet", index=False)

    # Fama-MacBeth-style linear baseline (validation checklist item 5,
    # feature_expansion_instructions.md) -- mirrors the GBM's exact
    # walk-forward fold structure so the two are directly comparable.
    # Reuses summarize_ic unmodified since run_linear_walk_forward
    # returns the identical [date, permno, fwd_ret, pred] shape.
    print("\n-- Linear (Fama-MacBeth) baseline IC summary --")
    linear_preds = run_linear_walk_forward(
        feature_panel, train_months=TRAIN_MONTHS, test_months=1,
        step_months=HORIZON, horizon=HORIZON,
    )
    linear_ic = summarize_ic(linear_preds)
    linear_ic.to_csv(OUTPUT_DIR / "ic_linear_baseline.csv", index=False)
    if ic["ic"].mean() <= linear_ic["ic"].mean():
        print("  FINDING: GBM does not beat the linear baseline on mean IC -- "
              "reporting as-is, not suppressing.")

    print("\n-- Linear baseline IC summary (top/bottom decile only) --")
    linear_ic_tails = summarize_tails_ic(linear_preds)
    linear_ic_tails.to_csv(OUTPUT_DIR / "ic_linear_baseline_tails.csv", index=False)

    plot_ic_timeseries({"gbm": ic}, OUTPUT_DIR / "ic_timeseries.png")
    plot_ic_timeseries({"gbm": ic, "baseline": linear_ic},
                        OUTPUT_DIR / "ic_timeseries_gbm_vs_baseline.png")

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
