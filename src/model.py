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
):
    """
    Generator yielding (train_dates, test_dates) tuples.

    Expanding-window alternative: set train_months=None to use all history
    up to the test window each time, instead of a fixed lookback. A fixed
    rolling window (as implemented here) is often preferred because it
    keeps the model from being trained on a regime that's no longer
    representative (e.g. pre-2020 volatility structure).

    NOTE ON PURGING: because mom_12m_ex1 uses a 12-month trailing window,
    a training observation dated one month before the test window
    technically has features overlapping the test period's information set
    only in edge cases (its 12m window doesn't reach into the test month
    itself, since it's already excluding month t-1). This implementation
    doesn't add an explicit embargo gap because the feature construction
    already avoids the overlap -- but if you add features with shorter
    lookback windows relative to your test window size, this is exactly
    where you'd insert a purge/embargo gap. Naming this reasoning in your
    write-up is the point, not just having a working generator.
    """
    unique_months = sorted(dates.unique())
    train_months = train_months or len(unique_months)

    i = train_months
    while i + test_months <= len(unique_months):
        train_window = unique_months[max(0, i - train_months):i]
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
) -> pd.DataFrame:
    """
    Runs the full walk-forward loop, training a fresh model on each window
    and generating predictions for the following out-of-sample month.

    model_type: "linear" (Fama-MacBeth-style baseline) or "gbm" (LightGBM).

    Returns a dataframe of out-of-sample predictions with columns:
    [date, permno, fwd_ret, pred] -- this is what backtest.py consumes.
    """
    results = []

    for train_window, test_window in walk_forward_splits(
        panel["date"], train_months=train_months, test_months=test_months
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
