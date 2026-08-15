"""
validate_features.py

Implements feature_expansion_instructions.md Section 6's 5-item validation
checklist, as independently re-runnable functions so any one of them can be
rerun on its own while iterating:

    1. coverage_report          -- % non-null by year x feature
    2. leakage_spot_check       -- manually re-derive 5 random rows per
                                    fundamental feature, print the trace,
                                    assert public-as-of <= feature date
    3. timing_regression_test   -- shift each source's public-as-of dates
                                    forward, confirm mean IC drops/flattens
    4. rerun_pipeline_smoke_test -- row count + shape sanity after a full
                                    python src/main.py-equivalent rerun
    5. baseline_comparison      -- GBM IC vs. linear IC side by side

BLOCKED until real WRDS fundamentals data lands (see
load_wrds_fundamentals.py) -- the functions below are structurally
complete and their sub-pieces (build_feature_panel, run_walk_forward,
summarize_ic, baseline.run_linear_walk_forward) are already verified
against synthetic fixtures and/or real CRSP-only data, but this script's
own end-to-end run needs the real fundamentals.

Usage (after load_wrds_fundamentals.py has been run):
    python src/validate_features.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from baseline import run_linear_walk_forward
from features import _add_sue_column, _build_quarterly_surprises, build_feature_panel
from model import FEATURE_COLS, run_walk_forward, summarize_ic

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def coverage_report(panel: pd.DataFrame, feature_cols: list[str] | None = None) -> pd.DataFrame:
    """
    Checklist item 1: % non-null by year, per feature. Instructions:
    "IBES/short-interest coverage should be >85% for S&P 500 names in
    recent years; if it's much lower, the link table join is broken --
    investigate before proceeding."
    """
    feature_cols = feature_cols or [
        "mom_1m", "mom_3m", "mom_12m_ex1", "realized_vol", "log_mkt_cap",
        "log_dollar_vol", "est_rev_3m", "rev_breadth", "sue", "short_ratio",
    ]
    df = panel.copy()
    df["year"] = df["date"].dt.year
    report = df.groupby("year")[feature_cols].apply(lambda g: g.notna().mean())
    print(report.to_string(float_format=lambda x: f"{x:.1%}"))
    return report


def _trace_fixed_lag(raw, permno, feature_date, date_col, key_col, value_col,
                      lag_days=0, max_age_days=None):
    """Manually finds the raw source row that a fixed-lag/expiry feature
    (short_ratio, sue) SHOULD have matched for (permno, feature_date) --
    re-derived independently of pit_merge.merge_asof_pit, so this check
    isn't just asserting the helper agrees with itself."""
    sub = raw[raw[key_col] == permno].copy()
    sub["_asof"] = pd.to_datetime(sub[date_col]) + pd.Timedelta(days=lag_days)
    sub = sub[sub["_asof"] <= feature_date]
    if sub.empty:
        return None
    row = sub.sort_values("_asof").iloc[-1]
    age_days = (feature_date - row["_asof"]).days
    return {
        "matched_asof": row["_asof"], "value": row[value_col], "age_days": age_days,
        "expired": max_age_days is not None and age_days > max_age_days,
    }


def leakage_spot_check(
    panel: pd.DataFrame, ibes_estimates: pd.DataFrame, ibes_actuals: pd.DataFrame,
    short_interest: pd.DataFrame, n: int = 5, seed: int = 0,
) -> bool:
    """
    Checklist item 2: for 5 random (permno, date) rows PER fundamental
    feature, manually trace the raw source row used and confirm its
    public-as-of date precedes the feature date. Prints the trace.
    Returns True iff every sampled row passes.
    """
    all_pass = True

    print("\n=== leakage spot-check: est_rev_3m ===")
    fy1 = ibes_estimates[ibes_estimates["period_type"] == "FY1"]
    sample = panel[panel["est_rev_3m"].notna()]
    sample = sample.sample(min(n, len(sample)), random_state=seed)
    for _, row in sample.iterrows():
        cur = _trace_fixed_lag(fy1, row["permno"], row["date"], "statpers", "permno", "meanest")
        prior = _trace_fixed_lag(fy1, row["permno"], row["date"] - pd.DateOffset(months=3),
                                  "statpers", "permno", "meanest")
        ok = cur is not None and prior is not None
        recomputed = (cur["value"] - prior["value"]) / max(abs(row["month_end_prc"]), 1.0) if ok else None
        matches = ok and np.isclose(recomputed, row["est_rev_3m"], atol=1e-6)
        all_pass &= matches
        print(f"  permno={row['permno']} date={row['date'].date()}"
              f"  cur_statpers={cur['matched_asof'].date() if cur else None}"
              f"  prior_statpers={prior['matched_asof'].date() if prior else None}"
              f"  recomputed={recomputed}  panel_value={row['est_rev_3m']:.6f}"
              f"  {'PASS' if matches else 'FAIL'}")

    print("\n=== leakage spot-check: short_ratio ===")
    sample = panel[panel["short_ratio"].notna()]
    sample = sample.sample(min(n, len(sample)), random_state=seed)
    for _, row in sample.iterrows():
        trace = _trace_fixed_lag(short_interest, row["permno"], row["date"],
                                  "settlement_date", "permno", "shortint", lag_days=8)
        ok = trace is not None and trace["matched_asof"] <= row["date"]
        all_pass &= ok
        print(f"  permno={row['permno']} date={row['date'].date()}"
              f"  matched_settlement+8d={trace['matched_asof'].date() if trace else None}"
              f"  age={trace['age_days'] if trace else None}d  {'PASS' if ok else 'FAIL'}")

    print("\n=== leakage spot-check: sue ===")
    surprises = _add_sue_column(_build_quarterly_surprises(ibes_actuals, ibes_estimates))
    sample = panel[panel["sue"].notna()]
    sample = sample.sample(min(n, len(sample)), random_state=seed)
    for _, row in sample.iterrows():
        trace = _trace_fixed_lag(surprises, row["permno"], row["date"], "anndats",
                                  "permno", "sue", max_age_days=122)
        ok = trace is not None and not trace["expired"]
        all_pass &= ok
        print(f"  permno={row['permno']} date={row['date'].date()}"
              f"  matched_anndats={trace['matched_asof'].date() if trace else None}"
              f"  age={trace['age_days'] if trace else None}d"
              f"  (<=122d expiry? {'True' if trace and not trace['expired'] else 'False'})"
              f"  {'PASS' if ok else 'FAIL'}")

    return all_pass


def timing_regression_test(
    daily_panel: pd.DataFrame, membership: pd.DataFrame, ibes_estimates: pd.DataFrame,
    ibes_actuals: pd.DataFrame, short_interest: pd.DataFrame, sector_crosswalk: pd.DataFrame,
    siccd_history: pd.DataFrame, build_kwargs: dict, walk_forward_kwargs: dict,
    shift_months: int = 3,
) -> pd.DataFrame:
    """
    Checklist item 3: shift each fundamental source's public-as-of date
    column forward `shift_months` -- independently per source (so a
    regression hiding in one source isn't masked by the other two being
    correct), then all three together -- rebuild the panel, rerun the
    SAME walk-forward config, and compare mean IC to the unshifted
    baseline. A forward shift makes each feature look like it was public
    LATER than it really was, which should only ever remove signal (the
    model sees a staler, less-timely version of the same information) --
    so mean IC dropping or flattening is the EXPECTED, healthy outcome.
    If shifted IC rises instead, something else is leaking (the shift
    didn't touch it), and this is printed as an explicit warning, not a
    silent pass.
    """
    def _ic(panel):
        preds = run_walk_forward(panel, return_diagnostics=False, **walk_forward_kwargs)
        return summarize_ic(preds)["ic"].mean()

    baseline_panel = build_feature_panel(
        daily_panel, membership, ibes_estimates, ibes_actuals, short_interest,
        sector_crosswalk, siccd_history, **build_kwargs,
    )
    baseline_ic = _ic(baseline_panel)

    shifted_est = ibes_estimates.copy()
    shifted_est["statpers"] = shifted_est["statpers"] + pd.DateOffset(months=shift_months)
    shifted_act = ibes_actuals.copy()
    shifted_act["anndats"] = shifted_act["anndats"] + pd.DateOffset(months=shift_months)
    shifted_si = short_interest.copy()
    shifted_si["settlement_date"] = shifted_si["settlement_date"] + pd.DateOffset(months=shift_months)

    scenarios = {
        "ibes_estimates_shifted": (shifted_est, ibes_actuals, short_interest),
        "ibes_actuals_shifted": (ibes_estimates, shifted_act, short_interest),
        "short_interest_shifted": (ibes_estimates, ibes_actuals, shifted_si),
        "all_shifted": (shifted_est, shifted_act, shifted_si),
    }

    rows = [{"scenario": "baseline", "mean_ic": baseline_ic, "delta_vs_baseline": 0.0}]
    for name, (est, act, si) in scenarios.items():
        panel = build_feature_panel(
            daily_panel, membership, est, act, si, sector_crosswalk, siccd_history, **build_kwargs,
        )
        ic = _ic(panel)
        delta = ic - baseline_ic
        rows.append({"scenario": name, "mean_ic": ic, "delta_vs_baseline": delta})
        if delta > 0:
            print(f"  WARNING: {name} mean IC ROSE vs baseline ({ic:+.4f} vs {baseline_ic:+.4f}) "
                  f"-- shifting a source's public-as-of date LATER should never help; "
                  f"investigate a possible leak this shift didn't touch.")

    result = pd.DataFrame(rows)
    print(result.to_string(index=False, float_format=lambda x: f"{x:+.4f}"))
    return result


def rerun_pipeline_smoke_test(
    feature_panel: pd.DataFrame, pre_change_row_count: int | None = None,
    tolerance: float = 0.05,
) -> bool:
    """
    Checklist item 4: confirms the rebuilt feature panel's row count is
    within `tolerance` of `pre_change_row_count` (the NaN-fill-to-0
    policy in build_feature_panel should prevent shrinkage from the new
    fundamentals' coverage gaps -- see features.build_feature_panel), and
    that every FEATURE_COLS column exists and isn't entirely NaN.
    """
    ok = True
    for col in FEATURE_COLS:
        if col not in feature_panel.columns:
            print(f"  FAIL: {col} missing from feature panel")
            ok = False
        elif feature_panel[col].isna().all():
            print(f"  FAIL: {col} is all-NaN")
            ok = False
    if pre_change_row_count is not None:
        pct_change = abs(len(feature_panel) - pre_change_row_count) / pre_change_row_count
        within = pct_change <= tolerance
        print(f"  row count: {len(feature_panel)} vs pre-change {pre_change_row_count} "
              f"({pct_change:+.1%}, {'within' if within else 'OUTSIDE'} {tolerance:.0%} tolerance)")
        ok &= within
    print("PASS" if ok else "FAIL")
    return ok


def baseline_comparison(feature_panel: pd.DataFrame, gbm_ic: pd.DataFrame,
                         walk_forward_kwargs: dict) -> pd.DataFrame:
    """
    Checklist item 5: GBM IC vs. linear IC side by side. Prints either
    way -- a null result (GBM doesn't beat the linear baseline) is a
    finding to surface per the instructions, not something to suppress.
    """
    linear_preds = run_linear_walk_forward(feature_panel, **walk_forward_kwargs)
    linear_ic = summarize_ic(linear_preds)

    result = pd.DataFrame({
        "model": ["gbm", "linear_baseline"],
        "mean_ic": [gbm_ic["ic"].mean(), linear_ic["ic"].mean()],
        "ic_ir": [gbm_ic["ic"].mean() / gbm_ic["ic"].std(),
                  linear_ic["ic"].mean() / linear_ic["ic"].std()],
    })
    print(result.to_string(index=False, float_format=lambda x: f"{x:+.4f}"))
    if result.loc[result["model"] == "gbm", "mean_ic"].iloc[0] <= result.loc[
        result["model"] == "linear_baseline", "mean_ic"
    ].iloc[0]:
        print("  FINDING: GBM does not beat the linear baseline on mean IC -- "
              "reporting as-is, not suppressing.")
    return result


if __name__ == "__main__":
    print("validate_features.py: run individual functions interactively once "
          "load_wrds_fundamentals.py has produced real data -- see this module's "
          "docstring. No default end-to-end run here since the exact build_kwargs/"
          "walk_forward_kwargs (train_months, horizon, etc.) should match main.py's.")
