"""
model.py

Walk-forward validation and model training. This is the module most likely
to get probed hard in an interview -- the discipline here (no random
splits, no shuffling across time) is the difference between a defensible
backtest and a leaked one.
"""

from __future__ import annotations

import pandas as pd
import numpy as np
from xgboost import XGBRanker
from scipy.stats import spearmanr, ttest_1samp
from tqdm import tqdm

from features import winsorize

# Default XGBRanker hyperparameters. Exposed as a module-level constant
# (not just inlined below) so tune.py's sweeps can start from
# `{**DEFAULT_MODEL_PARAMS, "max_depth": 3}` etc. rather than needing to
# restate every other value. objective="rank:ndcg" is XGBoost's closest
# analog to LightGBM's lambdarank -- both are LambdaMART-style,
# NDCG-optimizing rank objectives. min_child_weight is NOT the same thing
# as LightGBM's min_child_samples (it thresholds summed Hessian, not a
# raw sample count) -- kept at the same numeric value as a starting
# point, not because the two are equivalent (see README known
# limitations). random_state is fixed so tune.py's sweeps are actually
# reproducible once subsample/colsample_bytree (Tier 1) introduce real
# randomness -- at the default subsample=1.0 this is a no-op, but without
# it, re-running an identical subsample<1 config would silently give a
# different result each time, which would defeat the point of logging
# sweeps for side-by-side comparison. colsample_bytree=0.7 is a Tier 1
# tuning result (see hyperparameter_approach.md / tune.py) -- a paired
# per-fold comparison against colsample_bytree=1.0 showed a consistent
# lean (mean paired IC delta +0.009, same direction at 0.6/0.7/0.8) but
# did NOT clear significance (t~1.4, ~58% of folds better). Set
# provisionally as a low-risk, plausibly-neutral-to-mildly-positive
# choice, not as a confirmed win -- max_depth, min_child_weight, and
# subsample were all swept too and showed no distinguishable effect at
# any value (every result within ~0.3 SE of every other). See
# output/tuning_results.csv for the full sweep.
DEFAULT_MODEL_PARAMS = dict(
    objective="rank:pairwise", n_estimators=100, max_depth=4,
    learning_rate=0.01, min_child_weight=5, colsample_bytree=0.7,
    verbosity=0, random_state=0,
)

FEATURE_COLS = [
    # mom_1m_z and log_dollar_vol_z dropped per feature_expansion_instructions.md
    # Section 3 -- 1-month reversal decays inside the 3-month holding window
    # (wrong horizon for HORIZON=3), and log_dollar_vol has near-zero dispersion
    # within S&P 500 large caps. Both are still computed in features.py, just
    # not fed to the model.
    "mom_3m_z", "mom_12m_ex1_z", "realized_vol_z", "log_mkt_cap_z",
    "est_rev_3m_z", "rev_breadth_z", "sue_z", "short_ratio_z",
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
        i += step_months # this is where we use step_months


def information_coefficient(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Spearman rank correlation between predicted and realized returns.
    This is the metric practitioners actually use to evaluate a ranking
    model -- it cares about ORDER, not magnitude, which is what matters
    for a long-short portfolio built on ranks.
    """
    if len(y_true) < 2:
        return np.nan
    ic, _ = spearmanr(y_pred, y_true) # returns the spearman rank correlation coefficient
    return ic




# this is running the walk_forward validation, all of them
def run_walk_forward(
    panel: pd.DataFrame,
    train_months: int = 60,
    test_months: int = 1,
    step_months: int = 1,
    horizon: int = 1,
    model_params: dict | None = None,
    return_diagnostics: bool = False,
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

    `model_params` overrides DEFAULT_MODEL_PARAMS for hyperparameter
    sweeps (see tune.py) -- e.g. {"max_depth": 3} keeps every other
    default and only changes depth. Unspecified keys keep their default.

    `return_diagnostics`: when True, ALSO returns a second dataframe, one
    row per walk-forward fold, with in-sample vs. out-of-sample IC and
    fold window sizes -- what a hyperparameter sweep needs to see a
    train/test IC gap directly instead of inferring overfitting from the
    OOS mean alone. Off by default so the normal single-dataframe return
    (what backtest.py and every existing caller expects) is unchanged.

    Returns a dataframe of out-of-sample predictions with columns:
    [date, permno, fwd_ret, pred] -- this is what backtest.py consumes.
    """
    
    # this is combining the dictionaries. The syntax is set up so that the second dict overrides the first if need be
    params = {**DEFAULT_MODEL_PARAMS, **(model_params or {})}

    # here, panel is the feature_panel
    # train_months = 60 is the rolling lookback window for training
    results = []
    diagnostics = []
    embargo_months = max(0, horizon - 1)

    # each train_window ends up being a list of Timestamp objects of the 60 months trailing months to train on
    # each test_window is a list with one Timestamp object, 3 months after final training month (based on test_month=1, horizon=3)
    # train_window = [Timestamp('2012-01-31 00:00:00'), Timestamp('2012-02-29 00:00:00'), ... , Timestamp('2016-10-31 00:00:00')]
    # test_window = [Timestamp('2017-01-31 00:00:00')]

    for fold, (train_window, test_window) in enumerate(tqdm(walk_forward_splits(
        panel["date"], train_months=train_months, test_months=test_months,
        step_months=step_months, embargo_months=embargo_months))):
        
        # these are pure rows from feature_panel with correct windows
        train = panel[panel["date"].isin(train_window)] # pulls rows from only train_window
        test = panel[panel["date"].isin(test_window)]

        # skip if training data is too little or there is no test window
        if len(train) < 100 or len(test) == 0:
            continue

        # Winsorize the TRAINING target only, cross-sectionally per month,
        # so a handful of extreme-return months don't dominate the fit --
        # into its OWN column, not overwriting TARGET_COL. That keeps the
        # true fwd_ret available below for in-sample IC, which needs to
        # evaluate against the same real returns the OOS side does, not a
        # robustified version of them (evaluation must never be computed
        # on winsorized returns, in-sample or out). Sorted by date so rows
        # sharing a date are contiguous -- GBM's per-date `group`
        # boundaries below require that.
        train = train.copy().sort_values("date") # this sorting is important for format keeping

        # we group by the date, only extract the fwd_ret (the label) and then
        # winsorize per cross section... the thresholds are different for each cross section
        # wait but does winsorizing necessary here if we are only doing ranks?
        train["winsorized_target"] = train.groupby("date")[TARGET_COL].transform(winsorize)
        #train["winsorized_target"] = train[TARGET_COL]

        # we are going to train on these cross-sectional vectors
        # they are stripped off PERMNO, purely the factors in each row
        X_train = train[FEATURE_COLS]

        # X_test is purely the factors -- fwd_ret for the output comes
        # straight from `test` below, not from a separate y_test.
        X_test = test[FEATURE_COLS]

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

        # turns each stock's winsorized fwd_ret into a decile label (0-9 within its own month)
        # XGBRanker is learning to predict the rank based on the factors, not the return
        # Basically, the model will say, based on these factors, this stock will be nth rank this month
        rank_label = train.groupby("date")["winsorized_target"].transform(
            lambda s: pd.qcut(s, 10, labels=False, duplicates="drop")
            # qcut is a quantile-based binning, and we are using 10 for deciles
            # labels=False makes it return ints 0~9 instead of interval objects like (0.01,0.05]
        ) # ultimately these are the targets

        # ndcg = Normalized Discounted Cumulative Gain. It is a metric used
        # to measure the quality of a ranked list of items, checking how
        # well an algorithm puts the most relevant results at the top. In
        # XGBoost, it is used as a ranking metric and optimization
        # objective (rank:ndcg) via the LambdaMART algorithm.
        model = XGBRanker(**params)
        model.fit(X_train, rank_label, group=train_group)
        preds = model.predict(X_test) # outputs a list of ranking scores. The relative ordering of ranking scores matter, not the absolute magnitudes themselves

        out = test[["date", "permno", TARGET_COL]].copy()
        out["pred"] = preds
        results.append(out)
        # eventually, results will be a list of dataframes each containing the ranking scores for each PERMNO on each date
        
        # we are calculating diagnostics for in sample
        if return_diagnostics:
            # In-sample IC: same model, scored on its OWN training rows,
            # against the true (unwinsorized) fwd_ret -- comparable
            # apples-to-apples with oos_ic below, computed the same way.
            in_sample = train[["date", TARGET_COL]].copy()
            in_sample["pred"] = model.predict(X_train) # using training data since we are calculating in sample
            in_sample_ic = (
                in_sample.groupby("date")
                .apply(lambda g: information_coefficient(g[TARGET_COL], g["pred"]),
                       include_groups=False) # really just calculating the spearman rank correlation coefficient
                .mean()
            )
            oos_ic = (
                out.groupby("date") # note this uses 'out' for oos
                .apply(lambda g: information_coefficient(g[TARGET_COL], g["pred"]),
                       include_groups=False)
                .mean()
            )
            diagnostics.append({
                "fold": fold,
                "train_start": train_window[0],
                "train_end": train_window[-1],
                "n_train_months": len(train_window),
                "n_train_rows": len(train),
                "test_start": test_window[0],
                "test_end": test_window[-1],
                "in_sample_ic": in_sample_ic,
                "oos_ic": oos_ic,
                "gap": in_sample_ic - oos_ic,
            })

    # predictions will be a data frame that contains all of the predictions from all of the walk-forward validation folds
    # results is a list of dataframes and each dataframe contains the predictions of that walk-forward validation fold
    predictions = pd.concat(results, ignore_index=True)

    if return_diagnostics:
        return predictions, pd.DataFrame(diagnostics)
    
    return predictions


# this is for out of sample
def summarize_ic(predictions: pd.DataFrame) -> pd.DataFrame:
    """Monthly IC time series plus mean, IC information ratio (mean/std),
    and a t-test of whether the mean IC is significantly different from
    zero (H0: mean IC = 0) -- this is a per-month Spearman IC's own
    p-value would only say whether a single month's cross-sectional
    correlation is distinguishable from noise, not whether the model's
    overall OOS skill is, which is the number that actually matters here.
    Like oos_ic_std/oos_ic_ir (see tune.py), this assumes the monthly ICs
    are independent -- only true at step_months=horizon (see
    run_walk_forward's docstring); overlapping test windows would inflate
    the t-stat the same way they deflate oos_ic_std."""

    # preds is a dataframe of out-of-sample predictions
    # from all of the aggregated iterations of walk-forward validations with columns: [date, permno, fwd_ret, pred]
    # we are creating
    monthly_ic = (
        predictions.groupby("date")
        .apply(lambda g: information_coefficient(g[TARGET_COL], g["pred"]), include_groups=False)
        .rename("ic")
        .reset_index()
    )

    ic_values = monthly_ic["ic"].dropna()
    t_stat, p_value = ttest_1samp(ic_values, popmean=0)
    # stashed on .attrs rather than changing the return shape, since
    # callers (main.py, visualize.plot_ic_timeseries) already depend on
    # summarize_ic returning a plain [date, ic] dataframe
    monthly_ic.attrs["t_stat"] = t_stat
    monthly_ic.attrs["p_value"] = p_value

    print(f"Mean IC: {monthly_ic['ic'].mean():.4f}")
    print(f"Median IC: {monthly_ic['ic'].median():.4f}")
    print(f"IC std:  {monthly_ic['ic'].std():.4f}")
    print(f"IC IR:   {monthly_ic['ic'].mean() / monthly_ic['ic'].std():.4f}")
    print(f"t-stat:  {t_stat:.4f}  (p={p_value:.4f}, n={len(ic_values)}, H0: mean IC = 0)")
    return monthly_ic
