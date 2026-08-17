"""
baseline.py

Fama-MacBeth-style linear baseline, required by feature_expansion_instructions.md's
validation checklist item 5: "run the Fama-MacBeth-style linear baseline on the
same new feature set... Report GBM IC vs. linear IC side by side -- if the GBM
doesn't beat the linear baseline, that is a finding to surface, not suppress."

No statsmodels dependency (not installed in this environment; not worth adding
for one OLS) -- coefficients/t-stats via the normal equations directly.

Two distinct things live here, answering two different questions:
- `fama_macbeth_coefs`/`summarize_fama_macbeth`: the classic in-sample
  Fama-MacBeth procedure -- one cross-sectional OLS per date, coefficients
  averaged across time -- answers "is this feature's relationship with
  fwd_ret significant across independent monthly cross-sections?"
- `run_linear_walk_forward`: an out-of-sample walk-forward analog to
  model.run_walk_forward, for a fair GBM-vs-linear IC comparison over the
  SAME folds -- answers "does the GBM actually beat a simple linear model
  out of sample?"
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from features import winsorize
from model import FEATURE_COLS, TARGET_COL, walk_forward_splits


def _ols_with_tstats(X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Adds an intercept column, solves via lstsq, returns (coefs, t_stats)
    with coefs[0]/t_stats[0] the intercept. Degenerate windows (n <= number
    of params, e.g. a thin monthly cross-section with lots of NaN features
    dropped) return all-NaN rather than raising -- fama_macbeth_coefs calls
    this once per date and a single degenerate month shouldn't crash the
    whole procedure; summarize_fama_macbeth already drops NaNs when
    averaging across months.
    """
    n, k = X.shape
    Xd = np.column_stack([np.ones(n), X])
    k1 = Xd.shape[1]
    if n <= k1:
        return np.full(k1, np.nan), np.full(k1, np.nan)

    # Scoped, not global: on this platform's BLAS backend, `@` on these
    # matrices spuriously raises divide/overflow/underflow RuntimeWarnings
    # even when the result is fully finite and correct -- verified
    # directly (resid had zero NaNs, sigma2 was a normal finite positive
    # number) against a real fold before adding this. Kept local to this
    # function so a genuinely bad value elsewhere in the codebase would
    # still warn normally.
    with np.errstate(divide="ignore", over="ignore", under="ignore", invalid="ignore"):
        coefs, _, _, _ = np.linalg.lstsq(Xd, y, rcond=None)
        resid = y - Xd @ coefs
        dof = n - k1
        sigma2 = (resid @ resid) / dof
        xtx_inv = np.linalg.pinv(Xd.T @ Xd)
        se = np.sqrt(np.clip(np.diag(xtx_inv) * sigma2, 0, None))
        t_stats = np.where(se > 0, coefs / se, np.nan)
    return coefs, t_stats


def fama_macbeth_coefs(panel: pd.DataFrame, feature_cols: list[str] = FEATURE_COLS) -> pd.DataFrame:
    """
    One in-sample cross-sectional OLS of fwd_ret on feature_cols PER DATE
    (the classic Fama-MacBeth first stage) -- not a single pooled fit
    across dates, so a coefficient's eventual significance (see
    summarize_fama_macbeth) reflects variation ACROSS independent monthly
    cross-sections, not within-month sample size alone.

    Returns one row per date: n, and coef_<name>/tstat_<name> for
    name in ["intercept"] + feature_cols.
    """
    names = ["intercept"] + list(feature_cols)
    rows = []
    for date, g in panel.groupby("date"):
        g = g.dropna(subset=list(feature_cols) + [TARGET_COL])
        coefs, tstats = _ols_with_tstats(g[feature_cols].to_numpy(), g[TARGET_COL].to_numpy())
        row = {"date": date, "n": len(g)}
        for name, c, t in zip(names, coefs, tstats):
            row[f"coef_{name}"] = c
            row[f"tstat_{name}"] = t
        rows.append(row)
    return pd.DataFrame(rows).sort_values("date").reset_index(drop=True)


def summarize_fama_macbeth(fm_coefs: pd.DataFrame, feature_cols: list[str] = FEATURE_COLS) -> pd.DataFrame:
    """
    Per feature: avg_monthly_tstat (the literal average of each month's
    own regression t-stat -- what the validation checklist's "average the
    coefficient t-stats" literally asks for) AND fm_tstat_of_mean_coef
    (the textbook Fama-MacBeth inference statistic: mean(coef) /
    (std(coef) / sqrt(T)), testing whether the AVERAGE monthly
    coefficient is significantly nonzero across time) -- reported side by
    side since they answer overlapping but distinct questions, and
    neither is obviously "the" right one to report alone.
    """
    rows = []
    for col in ["intercept"] + list(feature_cols):
        coefs = fm_coefs[f"coef_{col}"].dropna()
        tstats = fm_coefs[f"tstat_{col}"].dropna()
        t = len(coefs)
        fm_t = coefs.mean() / (coefs.std() / np.sqrt(t)) if t > 1 and coefs.std() > 0 else np.nan
        rows.append({
            "feature": col,
            "mean_coef": coefs.mean(),
            "avg_monthly_tstat": tstats.mean(),
            "fm_tstat_of_mean_coef": fm_t,
            "n_months": t,
        })
    return pd.DataFrame(rows)


def run_linear_walk_forward(
    panel: pd.DataFrame,
    train_months: int = 60,
    test_months: int = 1,
    step_months: int = 3,
    horizon: int = 3,
    feature_cols: list[str] = FEATURE_COLS,
) -> pd.DataFrame:
    """
    Out-of-sample linear counterpart to model.run_walk_forward -- mirrors
    its EXACT fold structure (same walk_forward_splits call, same
    train/test extraction, same embargo) so folds line up 1:1 with the
    GBM's, making the two directly comparable rather than evaluated over
    different date ranges. Fits ONE pooled OLS per training window (not a
    per-month fit averaged, which would just be fama_macbeth_coefs) --
    the natural linear analog of what the GBM does per fold: one model
    per window, scored on the following OOS period. Training target is
    winsorized per month, same as model.run_walk_forward's GBM training
    target, for a fair comparison (OLS is at least as outlier-sensitive
    as a rank objective, arguably more so).

    Returns [date, permno, fwd_ret, pred] -- IDENTICAL shape to
    model.run_walk_forward's output, so model.summarize_ic() runs on it
    unmodified.
    """
    embargo_months = max(0, horizon - 1)
    results = []
    for train_window, test_window in walk_forward_splits(
        panel["date"], train_months=train_months, test_months=test_months,
        step_months=step_months, embargo_months=embargo_months,
    ):
        train = panel[panel["date"].isin(train_window)]
        test = panel[panel["date"].isin(test_window)]
        if len(train) < 100 or len(test) == 0:
            continue

        train = train.copy()
        train["winsorized_target"] = train.groupby("date")[TARGET_COL].transform(winsorize)

        coefs, _ = _ols_with_tstats(
            train[feature_cols].to_numpy(), train["winsorized_target"].to_numpy()
        )
        X_test = np.column_stack([np.ones(len(test)), test[feature_cols].to_numpy()])
        # same scoped BLAS-quirk suppression as _ols_with_tstats -- see its comment
        with np.errstate(divide="ignore", over="ignore", under="ignore", invalid="ignore"):
            pred = X_test @ coefs

        out = test[["date", "permno", TARGET_COL]].copy()
        out["pred"] = pred
        results.append(out)

    return pd.concat(results, ignore_index=True)
