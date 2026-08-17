"""
pit_merge.py

One reusable point-in-time asof-merge helper, used by every fundamental
feature added in features.py (analyst revisions, SUE, short interest).
Every fundamental merge in this pipeline MUST go through merge_asof_pit
-- it's the one place "was this number PUBLIC by the feature date?" is
enforced, so a timing mistake shows up here instead of being silently
re-implemented (and possibly re-broken) three separate times.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def merge_asof_pit(
    left: pd.DataFrame,
    right: pd.DataFrame,
    left_on: str,
    right_on: str,
    by: str,
    value_cols: list[str],
    lag_days: int = 0,
    max_age_days: int | None = None,
    suffix: str = "",
) -> pd.DataFrame:
    """
    Point-in-time asof-join of a fundamental source onto a monthly panel.

    - `right`'s `right_on` date is shifted forward by `lag_days` calendar
      days before matching -- this becomes the source's true public-as-of
      date (e.g. lag_days=8 for short-interest dissemination lag).
    - direction="backward": for each `left_on` date, matches the most
      recent `right` row whose public-as-of date is <= left_on. A future
      row can never match -- this is the actual leakage guard.
    - if `max_age_days` is set, matches older than that are nulled back
      to NaN (SUE's ~4-month expiry; None for IBES consensus / short
      interest, which just carry forward indefinitely).
    - `suffix` lets the same source be merged in twice under different
      names (est_rev_3m needs the SAME source matched at two different
      reference dates -- see features.add_analyst_revision_features).

    Two non-obvious pandas gotchas this function exists to get right
    once, verified empirically against pandas 2.2 (see the plan this was
    built from for a synthetic-fixture check):

    1. `pd.merge_asof(..., by=...)` requires the `on` column sorted
       GLOBALLY, not just within each `by` group -- sorting by
       `[by, left_on]` composite (the naive-looking approach) raises
       "ValueError: left keys must be sorted" as soon as `by` groups
       interleave in date order, which they always do in a stacked
       permno x date panel.
    2. `pd.merge_asof` always returns a FRESH RangeIndex, discarding
       whatever index `left` had, even though it preserves `left`'s row
       order exactly. Callers that combine two separate `merge_asof_pit`
       results via index-aligned Series arithmetic (again,
       add_analyst_revision_features) need the original index restored,
       or rows silently misalign with no error -- so this function
       restores it before returning.
    """
    left_sorted = left.sort_values(left_on)
    right_sorted = right[[by, right_on] + value_cols].dropna(subset=[right_on]).copy()
    right_sorted["_asof"] = right_sorted[right_on] + pd.Timedelta(days=lag_days)
    right_sorted = right_sorted.sort_values("_asof")

    merged = pd.merge_asof(
        left_sorted, right_sorted, left_on=left_on, right_on="_asof",
        by=by, direction="backward",
    )
    # merge_asof preserves left_sorted's row order but drops its index --
    # restore it so callers can combine multiple merge_asof_pit outputs
    # via label-aligned Series ops.
    merged.index = left_sorted.index

    if max_age_days is not None:
        age = (merged[left_on] - merged["_asof"]).dt.days
        merged.loc[age > max_age_days, value_cols] = np.nan

    merged = merged.drop(columns=["_asof", right_on], errors="ignore")
    if suffix:
        merged = merged.rename(columns={c: f"{c}{suffix}" for c in value_cols})

    # undo the internal on-sort, restore original row order
    return merged.sort_index()
