"""
model.py

Walk-forward validation and model training. This is the module most likely
to get probed hard in an interview -- the discipline here (no random
splits, no shuffling across time) is the difference between a defensible
backtest and a leaked one.
"""

import pandas as pd
import numpy as np
from xgboost import XGBRanker
from scipy.stats import spearmanr
from tqdm import tqdm

from features import winsorize


FEATURE_COLS = [
    "mom_1m_z", "mom_3m_z", "mom_12m_ex1_z", "realized_vol_z",
    "log_mkt_cap_z", "log_dollar_vol_z",
]
TARGET_COL = "fwd_ret"

# creates the (train_dates, test_dates) tuples that 
# each are a list of TimeStamps that we want to train and test on
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
    
    train_months = train_months or len(unique_months) # way to set default if train_months is non-zero/NaN/Empty

    i = train_months
    while i + test_months <= len(unique_months):
        train_end = max(0, i - embargo_months)
        train_start = max(0, train_end - train_months)
        train_window = unique_months[train_start:train_end]
        test_window = unique_months[i:i + test_months]
        yield train_window, test_window # this is what makes it a generator object
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

# this is really the train() method but in a walk-forward fashion
def run_walk_forward(
    panel: pd.DataFrame,
    train_months: int = 60,
    test_months: int = 1,
    step_months: int = 1,
    horizon: int = 1,
) -> pd.DataFrame:
    """
    Runs the full walk-forward loop, training a fresh XGBoost ranker on
    each window and generating predictions for the following out-of-sample
    period.

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
    
    # here, panel is the feature_panel
    # train_months = 60 is the rolling lookback window for training
    results = []
    embargo_months = max(0, horizon - 1)
    
    # each train_window ends up being a list of Timestamp objects of the 60 months trailing months to train on
    # each test_window is a list of Timestamp object, 3 months after final training month (based on test_month=1, horizon=3)
    
    # So however many sliding windows are available in 
    for train_window, test_window in tqdm(walk_forward_splits(
        panel["date"], train_months=train_months, test_months=test_months,
        step_months=step_months, embargo_months=embargo_months)):
        
        # these are pure rows from feature_panel with correct windows
        train = panel[panel["date"].isin(train_window)] # pulls rows from only train_window
        test = panel[panel["date"].isin(test_window)]
        
        # skip if training data is too little or there is no test window
        if len(train) < 100 or len(test) == 0:
            continue

        # Winsorize the TRAINING target only, cross-sectionally per month,
        # so a handful of extreme-return months don't dominate the fit.
        # Evaluation still uses the true fwd_ret in y_test (below) -- IC and
        # backtest performance are never computed on winsorized returns.
        # Sorted by date so that rows sharing a date are contiguous -- GBM's
        # per-date `group` boundaries below require that.
        train = train.copy().sort_values("date")
        
        # we group by the date, only extract the fwd_ret (the label) and then
        # winsorize per cross section... the thresholds are different for each cross section
        # we are overriding previous value with the clamped (winsorized) values
        train[TARGET_COL] = train.groupby("date")[TARGET_COL].transform(winsorize)
        
        # we are going to train on these cross-sectional vectors
        # they are stripped off PERMNO, purely the factors in each row
        X_train = train[FEATURE_COLS]
        
        # X_test also is just purely the factors
        # y_test is purely the fwd_ret
        X_test, y_test = test[FEATURE_COLS], test[TARGET_COL]

        # Plain L2 regression on fwd_ret optimizes predicted RETURN
        # MAGNITUDE, pooled across the whole training window -- but IC and
        # the decile long-short portfolio only care about rank ORDER within
        # each month. A rank objective optimizes that directly: the training
        # label is each stock's fwd_ret DECILE within its own date (matching
        # the deciles backtest.py actually trades), and `group` tells
        # XGBoost where one date's cross-section ends and the next begins so
        # ranking pairs are never compared across dates.

        # train_group is an array containing the number of stocks per each training month
        # we do this to tell XGBoost where one month's cross-section ends
        # and the next begins, so a stock is only ever compared against
        # its own month's peers — never ranked against a different month.
        train_group = train.groupby("date").size().to_numpy()

        # turns each stock's continuous fwd_ret into a decile label (0-9 within its own month)
        # XGBRanker is learning to predict the rank based on the factors, not the return
        # Basically, the model will say, based on these factors, this stock will be nth rank this month
        rank_label = train.groupby("date")[TARGET_COL].transform(
            lambda s: pd.qcut(s, 10, labels=False, duplicates="drop")
            # qcut is a quantile-based binning, and we are using 10 for deciles
            # labels=False makes it return ints 0~9 instead of interval objects like (0.01,0.05]
        ) # ultimately these are the targets

        # objective="rank:ndcg" is XGBoost's closest analog to LightGBM's
        # lambdarank -- both are LambdaMART-style, NDCG-optimizing rank
        # objectives. min_child_weight is NOT the same thing as LightGBM's
        # min_child_samples (it thresholds summed Hessian, not a raw sample
        # count) -- kept at the same numeric value as a starting point, not
        # because the two are equivalent (see README known limitations).
        model = XGBRanker(
            objective="rank:ndcg", n_estimators=100, max_depth=5,
            learning_rate=0.01, min_child_weight=30, verbosity=0,
        )
        model.fit(X_train, rank_label, group=train_group)

        preds = model.predict(X_test)

        out = test[["date", "permno", TARGET_COL]].copy()
        out["pred"] = preds
        results.append(out)
        
    # results is a list that contains M entries where M is the number of train/test windows
    # and each entry is a month of predictions (HORIZON=3) months out past the training date

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
