"""
model.py

Walk-forward validation and model training. This is the module most likely
to get probed hard in an interview -- the discipline here (no random
splits, no shuffling across time) is the difference between a defensible
backtest and a leaked one.
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from scipy.stats import spearmanr

from features import winsorize

try:
    from lightgbm import LGBMRegressor
    HAS_LGBM = True
except ImportError:
    HAS_LGBM = False
    print("lightgbm not installed -- pip install lightgbm to use the GBM model")


FEATURE_COLS = [
    "mom_1m_z", "mom_3m_z", "mom_12m_ex1_z", "realized_vol_z",
    "log_mkt_cap_z", "log_dollar_vol_z",
]
TARGET_COL = "fwd_ret"


def walk_forward_splits(
    dates: pd.Series,
    train_months: int = 60,
    test_months: int = 1,
    step_months: int = 1,
    embargo_months: int = 0,
):
    """
    Generator yielding (train_dates, test_dates) tuples.

    Expanding-window alternative: set train_months=None to use all history
    up to the test window each time, instead of a fixed lookback. A fixed
    rolling window (as implemented here) is often preferred because it
    keeps the model from being trained on a regime that's no longer
    representative (e.g. pre-2020 volatility structure).

    NOTE ON PURGING: with a 1-month-forward target, a training observation
    dated one month before the test window doesn't overlap the test
    period's information set (mom_12m_ex1's 12m lookback doesn't reach
    forward into the test month). That assumption breaks once the target
    itself looks further forward than 1 month: a training row's fwd_ret is
    only "known" `horizon` months after its feature date, so any training
    row within `horizon - 1` months of the test start has a label that
    wouldn't actually be realized yet at test time -- using it anyway is
    leakage. `embargo_months` drops exactly those trailing training months
    (run_walk_forward sets it to `horizon - 1` automatically). At
    embargo_months=0 (the 1-month-horizon default) this is a no-op and
    behaves exactly as before.
    """
    unique_months = sorted(dates.unique())
    train_months = train_months or len(unique_months)

    i = train_months
    while i + test_months <= len(unique_months):
        train_end = max(0, i - embargo_months)
        train_start = max(0, train_end - train_months)
        train_window = unique_months[train_start:train_end]
        test_window = unique_months[i:i + test_months]
        yield train_window, test_window
        i += step_months


def information_coefficient(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Spearman rank correlation between predicted and realized returns.
    This is the metric practitioners actually use to evaluate a ranking
    model -- it cares about ORDER, not magnitude, which is what matters
    for a long-short portfolio built on ranks.
    """
    if len(y_true) < 2:
        return np.nan
    ic, _ = spearmanr(y_pred, y_true)
    return ic


def run_walk_forward(
    panel: pd.DataFrame,
    model_type: str = "linear",
    train_months: int = 60,
    test_months: int = 1,
    step_months: int = 1,
    horizon: int = 1,
) -> pd.DataFrame:
    """
    Runs the full walk-forward loop, training a fresh model on each window
    and generating predictions for the following out-of-sample period.

    model_type: "linear" (Fama-MacBeth-style baseline) or "gbm" (LightGBM).

    `horizon` must match the horizon used to build `fwd_ret` in
    features.build_feature_panel -- it's used to embargo training labels
    that wouldn't actually be known yet at test time (see
    walk_forward_splits' NOTE ON PURGING). When horizon > 1, keep
    test_months=1 (one evaluation snapshot per step) and set
    step_months=horizon, so consecutive test dates are exactly `horizon`
    months apart and their fwd_ret windows are adjacent, not overlapping.
    Setting test_months=horizon instead does NOT achieve this -- every
    calendar month inside that wider test window would still get its own
    overlapping horizon-month-forward label. Non-overlapping return
    observations are required for backtest.py to validly compound them as
    a sequential return series.

    Returns a dataframe of out-of-sample predictions with columns:
    [date, permno, fwd_ret, pred] -- this is what backtest.py consumes.
    """
    results = []
    embargo_months = max(0, horizon - 1)

    for train_window, test_window in walk_forward_splits(
        panel["date"], train_months=train_months, test_months=test_months,
        step_months=step_months, embargo_months=embargo_months,
    ):
        train = panel[panel["date"].isin(train_window)]
        test = panel[panel["date"].isin(test_window)]

        if len(train) < 100 or len(test) == 0:
            continue

        # Winsorize the TRAINING target only, cross-sectionally per month,
        # so a handful of extreme-return months don't dominate the fit.
        # Evaluation still uses the true fwd_ret in y_test (below) -- IC and
        # backtest performance are never computed on winsorized returns.
        train = train.copy()
        train[TARGET_COL] = train.groupby("date")[TARGET_COL].transform(winsorize)

        X_train, y_train = train[FEATURE_COLS], train[TARGET_COL]
        X_test, y_test = test[FEATURE_COLS], test[TARGET_COL]

        if model_type == "linear":
            model = LinearRegression()
        elif model_type == "gbm":
            if not HAS_LGBM:
                raise ImportError("lightgbm not installed")
            model = LGBMRegressor(
                n_estimators=200, max_depth=4, learning_rate=0.05,
                min_child_samples=30, verbosity=-1,
            )
        else:
            raise ValueError(f"Unknown model_type: {model_type}")

        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        out = test[["date", "permno", TARGET_COL]].copy()
        out["pred"] = preds
        results.append(out)

    return pd.concat(results, ignore_index=True)


def summarize_ic(predictions: pd.DataFrame) -> pd.DataFrame:
    """Monthly IC time series plus mean and IC information ratio (mean/std)."""
    monthly_ic = (
        predictions.groupby("date")
        .apply(lambda g: information_coefficient(g[TARGET_COL], g["pred"]))
        .rename("ic")
        .reset_index()
    )
    print(f"Mean IC: {monthly_ic['ic'].mean():.4f}")
    print(f"IC std:  {monthly_ic['ic'].std():.4f}")
    print(f"IC IR:   {monthly_ic['ic'].mean() / monthly_ic['ic'].std():.4f}")
    return monthly_ic
