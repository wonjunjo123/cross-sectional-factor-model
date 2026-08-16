"""
features.py

Builds cross-sectional factors from the CRSP daily price panel, resampled
to a monthly frequency (standard for this kind of factor model -- daily
rebalancing isn't realistic or common in cross-sectional equity research).

KEY PRINCIPLE #1 (look-ahead): every feature must be computed using only
information available AS OF the observation date. The most common place
this bites: computing a rolling window feature (e.g. 12-month momentum)
using data up to and including month-end t, then using it to predict
returns for month t itself instead of month t+1. Always predict FORWARD
from the feature date.

KEY PRINCIPLE #2 (survivorship): features are computed across each
PERMNO's FULL available price history first (a stock needs trailing price
data to compute momentum even in months just before or after its actual
index membership window). Point-in-time S&P 500 membership is applied as
a FILTER afterward, right before cross-sectional normalization -- so the
cross-section on any given date only ever contains stocks that were
genuinely index members on that date, without truncating the trailing
history needed to compute their features correctly. Filtering too early
throws away legitimate lookback data; filtering too late lets
non-member-on-that-date stocks leak into the cross-sectional ranking.
"""

from __future__ import annotations

import pandas as pd
import numpy as np

from pit_merge import merge_asof_pit


def resample_to_monthly(panel: pd.DataFrame) -> pd.DataFrame:
    """
    Collapses the daily CRSP panel to month-end observations per PERMNO.
    Cumulative return index, month-end market cap, average dollar volume trade, daily std for that monthff

    Builds a cumulative return index per PERMNO from CRSP's `ret` (total
    return, dividend- and delisting-adjusted by CRSP itself) rather than
    from a price series -- this is the correct way to compute momentum on
    CRSP data, since `ret` already embeds all the adjustments that the
    earlier yfinance version had to approximate via adjusted close.
    """
    panel = panel.sort_values(["permno", "date"]).copy()
    
    #cum_ret_index is a column that says “up until this date, this stock has had this growth factor (1 + ret)
    panel["cum_ret_index"] = panel.groupby("permno")["ret"].apply(
        lambda r: (1 + r.fillna(0)).cumprod() # r is a series of returns for that permno
    ).reset_index(level=0, drop=True)
    
    # create a new column just for month so that we can aggregate by it
    panel["month"] = panel["date"].dt.to_period("M")

    monthly = (
        panel.groupby(["permno", "month"])
        .agg(
            cum_ret_index=("cum_ret_index", "last"),
            mkt_cap=("mkt_cap", "last"),
            avg_dollar_vol=("dollar_vol", "mean"),
            daily_ret_std=("ret", "std"),  # realized vol for this month
            month_end_prc=("prc", "last"),  # est_rev_3m's price denominator
            shrout=("shrout", "last"),  # short_ratio's share-count denominator
        )
        .reset_index()
    )
    
    # just round date cleanly to month
    monthly["date"] = monthly["month"].dt.to_timestamp("M")
    
    # we were only using the month column to aggregate so now we drop it
    return monthly.drop(columns="month").sort_values(["date","permno"])


def add_momentum_features(monthly: pd.DataFrame) -> pd.DataFrame:
    """
    Standard momentum factors, computed PER PERMNO via groupby (never loop
    over securities manually -- groupby+shift/pct_change is vectorized and
    orders of magnitude faster on a panel this size).

    12-1 momentum: 12-month return EXCLUDING the most recent month. This
    is deliberate, not an off-by-one: 1-month reversal is a distinct,
    opposite-signed effect from 12-month momentum. Mixing them muddies
    the signal.
    """
    monthly = monthly.sort_values(["permno", "date"]).copy()
    g = monthly.groupby("permno")["cum_ret_index"]

    monthly["mom_1m"] = g.pct_change(1)
    monthly["mom_3m"] = g.pct_change(3)
    # this is the return for the past 11 months starting a year ago, so excludes this month.
    monthly["mom_12m_ex1"] = g.shift(1) / g.shift(12) - 1

    return monthly


def add_volatility_feature(monthly: pd.DataFrame) -> pd.DataFrame:
    """Trailing realized daily-return volatility, already aggregated in resample step.
        This is solely renaming"""
        
    return monthly.rename(columns={"daily_ret_std": "realized_vol"})


def add_size_and_liquidity_features(monthly: pd.DataFrame) -> pd.DataFrame:
    """
    log(market cap): a TRUE size factor now, using CRSP shares outstanding
    -- this replaces the log(dollar volume) stand-in from the yfinance
    version, which was flagged there explicitly as not being a real size
    proxy. Liquidity is kept as a separate feature (log dollar volume) so
    size and liquidity aren't conflated into one factor.
    """
    monthly["log_mkt_cap"] = np.log(monthly["mkt_cap"].clip(lower=1))
    monthly["log_dollar_vol"] = np.log(monthly["avg_dollar_vol"].clip(lower=1))
    return monthly


def add_analyst_revision_features(monthly: pd.DataFrame, ibes_estimates: pd.DataFrame) -> pd.DataFrame:
    """
    est_rev_3m = (current FY1 consensus mean EPS - FY1 consensus mean
    3 months ago) / abs(month-end price, floored at $1 so the
    denominator can't blow up near zero). rev_breadth = (numup -
    numdown) / numest (numest floored at 1, same reason).

    Timing: IBES STATPERS is the public-as-of date for a consensus --
    usable the same month, no extra lag needed (unlike SUE, which has to
    wait for an announcement). "3 months ago" is matched via its OWN
    independent backward asof against (date - 3mo), not by shifting the
    current match -- so it can never accidentally pull a future STATPERS
    either; each of the two matches is independently leak-safe.
    """
    fy1 = ibes_estimates[ibes_estimates["period_type"] == "FY1"]

    ref = monthly[["permno", "date"]].copy()
    ref["_ref"] = ref["date"]
    cur = merge_asof_pit(ref, fy1, "_ref", "statpers", "permno",
                          ["meanest", "numest", "numup", "numdown"], suffix="_cur")

    ref["_ref"] = ref["date"] - pd.DateOffset(months=3)
    prior = merge_asof_pit(ref, fy1, "_ref", "statpers", "permno",
                            ["meanest"], suffix="_3m_ago")

    monthly = monthly.copy()
    price = monthly["month_end_prc"].abs().clip(lower=1.0)
    monthly["est_rev_3m"] = (cur["meanest_cur"] - prior["meanest_3m_ago"]) / price
    monthly["rev_breadth"] = (
        (cur["numup_cur"] - cur["numdown_cur"]) / cur["numest_cur"].clip(lower=1)
    )
    return monthly


def _build_quarterly_surprises(ibes_actuals: pd.DataFrame, ibes_estimates: pd.DataFrame) -> pd.DataFrame:
    """
    One row per (permno, fpedats) reported quarter, joined to the LAST
    quarterly consensus published strictly before that quarter's
    ANNDATS -- matched by BOTH permno and fpedats (via merge_asof's
    `by=[...]` support for multiple keys), not just the closest-in-time
    consensus regardless of fiscal period, since IBES keeps revising a
    NOT-yet-reported quarter's estimate right up until the print, and a
    plain nearest-in-time match could accidentally grab the wrong
    quarter's consensus.
    """
    qtr = ibes_estimates[ibes_estimates["period_type"] == "QTR"]
    qtr = qtr.rename(columns={"statpers": "_asof"}).sort_values("_asof")
    act = ibes_actuals.sort_values("anndats")

    merged = pd.merge_asof(
        act, qtr, left_on="anndats", right_on="_asof",
        by=["permno", "fpedats"], direction="backward",
    )
    merged["surprise"] = merged["actual"] - merged["meanest"]
    return merged.sort_values(["permno", "fpedats"]).reset_index(drop=True)


def _add_sue_column(surprises: pd.DataFrame) -> pd.DataFrame:
    """
    Adds the 3-tier `sue` column to a quarterly surprises table (see
    add_sue_feature's docstring for the tiering rule). Split out from
    add_sue_feature so validate_features.py's leakage spot-check can
    recompute the exact same sue value when tracing a row, instead of
    re-implementing the tiering logic a second time.
    """
    surprises = surprises.copy()
    g = surprises.groupby("permno")["surprise"]
    prior_std = g.transform(lambda s: s.shift(1).rolling(8, min_periods=8).std())
    n_prior = g.transform(lambda s: s.shift(1).rolling(8, min_periods=1).count())

    surprises["sue"] = np.select(
        [n_prior >= 8, n_prior >= 4],
        [surprises["surprise"] / prior_std,
         surprises["surprise"] / surprises["stdev"].clip(lower=1e-4)],
        default=np.nan,
    )
    return surprises


def add_sue_feature(
    monthly: pd.DataFrame, ibes_actuals: pd.DataFrame, ibes_estimates: pd.DataFrame
) -> pd.DataFrame:
    """
    sue = (actual - last pre-announcement consensus mean) / denominator,
    a 3-tier denominator resolving an apparent tension in the source
    spec between "fall back if surprise history is short" and "require
    >=4 prior quarters": >=8 prior quarters -> own rolling std of the
    PRIOR 8 quarterly surprises (excludes the current one via
    .shift(1)); 4-7 prior quarters -> the cross-analyst consensus STDEV
    (more stable than a std computed on <8 points); <4 prior quarters ->
    NaN (hard gate, per the literal "require >=4"). See _add_sue_column.

    Timing: becomes public on ANNDATS, not fiscal period end. Persists
    until the next announcement; expires to NaN once >4 months old
    (max_age_days=122, a day-count approximation of "4 months" --
    exact to within a day or two, immaterial at this panel's monthly
    granularity).
    """
    surprises = _add_sue_column(_build_quarterly_surprises(ibes_actuals, ibes_estimates))
    return merge_asof_pit(monthly, surprises, "date", "anndats", "permno",
                           ["sue"], max_age_days=122)


def add_short_interest_feature(monthly: pd.DataFrame, short_interest: pd.DataFrame) -> pd.DataFrame:
    """
    short_ratio = shortint / shares_outstanding, shares from the SAME
    CRSP `shrout` field mkt_cap is built from (resample_to_monthly:
    mkt_cap = prc * shrout * 1000, shrout in thousands) -- kept
    unit-consistent with that construction rather than pulling shares
    from a separate Compustat source.

    Timing: settlement date + 8 calendar days (dissemination lag) must
    be <= month-end t -- `short_interest` must carry a `settlement_date`
    column (see load_wrds_fundamentals.py).
    """
    monthly = merge_asof_pit(monthly, short_interest, "date", "settlement_date",
                              "permno", ["shortint"], lag_days=8)
    monthly["short_ratio"] = monthly["shortint"] / (monthly["shrout"].clip(lower=1) * 1000)
    return monthly.drop(columns=["shortint"])


# Fama-French 12-industry SIC-code ranges, used by attach_sector's fallback
# path for PERMNOs with no valid CCM/GICS link on a given date. Sourced
# from Ken French's published "Detail for 12 Industry Portfolios"
# definitions (Siccodes12), cross-checked against a maintained open
# replication (github.com/ed-dehaan/FamaFrenchIndustries) rather than
# transcribed from memory -- a wrong boundary here silently misclassifies
# names into the wrong sector-neutral cohort, which would be invisible in
# any single downstream number. Worth a final check against Ken French's
# own zip (Siccodes12.zip) before relying on this in a published result.
_FF12_RANGES = [
    ("NoDur", [(100, 999), (2000, 2399), (2700, 2749), (2770, 2799), (3100, 3199), (3940, 3989)]),
    ("Durbl", [(2500, 2519), (2590, 2599), (3630, 3659), (3710, 3711), (3714, 3714),
               (3716, 3716), (3750, 3751), (3792, 3792), (3900, 3939), (3990, 3999)]),
    ("Manuf", [(2520, 2589), (2600, 2699), (2750, 2769), (3000, 3099), (3200, 3569),
               (3580, 3629), (3700, 3709), (3712, 3713), (3715, 3715), (3717, 3749),
               (3752, 3791), (3793, 3799), (3830, 3839), (3860, 3899)]),
    ("Enrgy", [(1200, 1399), (2900, 2999)]),
    ("Chems", [(2800, 2829), (2840, 2899)]),
    ("BusEq", [(3570, 3579), (3660, 3692), (3694, 3699), (3810, 3829), (7370, 7379)]),
    ("Telcm", [(4800, 4899)]),
    ("Utils", [(4900, 4949)]),
    ("Shops", [(5000, 5999), (7200, 7299), (7600, 7699)]),
    ("Hlth", [(2830, 2839), (3693, 3693), (3840, 3859), (8000, 8099)]),
    ("Money", [(6000, 6999)]),
    # Anything not covered by the 11 named ranges above (including a
    # missing/unparseable siccd) falls through to "other" -- see
    # _ff12_from_siccd's default -- matching the instructions' explicit
    # "map to a residual other bucket, don't drop."
]


def _ff12_from_siccd(siccd: pd.Series) -> pd.Series:
    """Vectorized SIC -> Fama-French-12-industry-bucket lookup. See
    _FF12_RANGES for the source and its caveat."""
    code = pd.to_numeric(siccd, errors="coerce")
    result = pd.Series("other", index=siccd.index, dtype=object)
    for label, ranges in _FF12_RANGES:
        mask = pd.Series(False, index=siccd.index)
        for lo, hi in ranges:
            mask |= code.between(lo, hi)
        result = result.mask(mask, label)
    return result


def attach_sector(
    monthly: pd.DataFrame, sector_crosswalk: pd.DataFrame, siccd_history: pd.DataFrame
) -> pd.DataFrame:
    """
    GICS sector via the CCM PERMNO<->GVKEY link VALID AS OF the feature
    date (LINKDT <= date <= LINKENDDT) -- not "whatever GVKEY this
    PERMNO currently maps to" -- so a PERMNO whose CCM link changed
    mid-sample (spinoff, corporate action) gets the sector that was
    actually linked at that date. PERMNOs with no valid link on a given
    date (link-table gap, or no CCM link at all) fall back to CRSP siccd
    -> Fama-French 12 industry instead of being dropped.

    Applied AFTER filter_to_membership (unlike the fundamentals above,
    which run before it) -- sector is a point-in-time snapshot with no
    trailing lookback window, so there's no "starved of legitimate
    history" risk from filtering first, and it's cheaper to join against
    the already-membership-filtered row count.

    `sector_crosswalk` columns: [permno, linkdt, linkenddt, gsector].
    `siccd_history` columns: [permno, sic_start, sic_end, siccd].

    Unlike merge_asof_pit, this returns the WHOLE panel wholesale rather
    than merging specific columns back onto an existing frame by index
    (the caller reassigns `monthly = attach_sector(monthly, ...)`
    entirely) -- so it only needs pd.merge_asof's row-order-preservation
    property, not index preservation. Each successive merge_asof stays
    sorted by `date` (the previous step's output order), so the second
    call doesn't need to re-sort.
    """
    m = monthly.sort_values("date").reset_index(drop=True)

    sc = sector_crosswalk.sort_values("linkdt")
    merged = pd.merge_asof(m, sc, left_on="date", right_on="linkdt",
                            by="permno", direction="backward")
    expired = merged["linkenddt"].notna() & (merged["date"] > merged["linkenddt"])
    merged.loc[expired, "gsector"] = np.nan

    sic = siccd_history.sort_values("sic_start")
    merged = pd.merge_asof(merged, sic, left_on="date", right_on="sic_start",
                            by="permno", direction="backward")
    sic_expired = merged["sic_end"].notna() & (merged["date"] > merged["sic_end"])
    merged.loc[sic_expired, "siccd"] = np.nan

    missing = merged["gsector"].isna()
    # Int64 (nullable) round-trip avoids "45.0" string artifacts from
    # gsector's float dtype (NaN-containing numeric columns are always
    # float in pandas) before stringifying for the grouping key.
    gsector_label = merged["gsector"].astype("Int64").astype(str)
    merged["sector"] = np.where(missing, np.nan, gsector_label)
    merged.loc[missing, "sector"] = _ff12_from_siccd(merged.loc[missing, "siccd"])

    return merged.drop(columns=["linkdt", "linkenddt", "gsector", "sic_start", "sic_end", "siccd"])


def add_forward_return_target(monthly: pd.DataFrame, horizon: int = 1) -> pd.DataFrame:
    """
    Target variable: forward return over `horizon` months, computed PER
    PERMNO with a negative shift (pulls the FUTURE value back to the
    current row). This is what the model predicts.
    """
    # this is the true fwd_ret value we compute
    # so that we can compute the true rankings to use as the target label
    # basically, we will be ranking the true 3-month-out forward return for a  so that
    # the rankings our model spits out will be inherently implying the 3-month-out forward return
    monthly = monthly.sort_values(["permno", "date"]).copy()
    monthly["fwd_ret"] = (
        monthly.groupby("permno")["cum_ret_index"].pct_change(horizon).shift(-horizon)
    )
    
    return monthly


def filter_to_membership(monthly: pd.DataFrame, membership: pd.DataFrame) -> pd.DataFrame:
    """
    THE survivorship-bias fix. Keeps only (permno, date) rows where `date`
    falls inside one of that permno's actual S&P 500 membership spells.

    Applied AFTER feature computation (see module docstring) so trailing-
    window features aren't starved of legitimate pre-membership history,
    but BEFORE cross-sectional normalization, so the cross-section used
    for z-scoring and the eventual portfolio only ever contains genuine
    point-in-time index members -- including names that have since been
    delisted, acquired, or dropped from the index.
    """
    merged = monthly.merge(membership, on="permno", how="inner")
    mask = (merged["date"] >= merged["start"]) & (merged["date"] <= merged["ending"])
    valid_keys = merged.loc[mask, ["permno", "date"]].drop_duplicates()

    filtered = monthly.merge(valid_keys, on=["permno", "date"], how="inner")
    return filtered


def winsorize(x: pd.Series, lower: float = 0.01, upper: float = 0.99) -> pd.Series:
    """
    Clips a series to its [lower, upper] quantiles. A handful of extreme
    values each month (e.g. an earnings-surprise return, a momentum spike
    right after a name re-enters the index) can otherwise dominate a
    cross-section's mean/std, distorting the z-score for every OTHER stock
    that month.
    """
    lo, hi = x.quantile(lower), x.quantile(upper)
    return x.clip(lower=lo, upper=hi)


def cross_sectional_normalize(
    monthly: pd.DataFrame,
    feature_cols: list[str],
    sector_col: str | None = None,
    min_group_size: int = 10,
) -> pd.DataFrame:
    """
    Z-scores every feature WITHIN each date's cross-section, not across
    the whole panel. This is the step that makes it a genuinely
    cross-sectional model: a momentum z-score of +1.5 means "1.5 std above
    the median INDEX MEMBER that month" -- and because this runs after
    filter_to_membership, "that month" now correctly excludes stocks that
    weren't actually in the index yet/anymore.

    Each feature is winsorized (see `winsorize`) within the same
    cross-section before the mean/std are computed, so the z-score itself
    isn't skewed by a few outliers before it's even used.

    `sector_col`: if given, z-scores within (date, sector_col) instead of
    date alone -- the point being to remove an unintended sector bet
    (e.g. "high momentum" ~= "is tech" in 2020) from EVERY feature, not
    just the new fundamentals. `sector_col=None` (the default) reproduces
    the exact prior date-only behavior.

    Guard: a (date, sector) cell with fewer than `min_group_size`
    NON-NULL observations of that SPECIFIC feature falls back to the
    full-date cross-section z-score for those rows, rather than
    producing a z-score from a handful of names. Measured per-feature via
    `transform("count")` on the feature column itself, not group row
    count -- a sector cell can have 15 names but only 6 non-null `sue`
    values (fundamentals have real coverage gaps), which should still
    fall back for `sue` even though it wouldn't for a fully-populated
    price feature in the same cell.
    """
    monthly = monthly.copy()

    def zscore(x):
        x = winsorize(x)
        std = x.std()
        # pd.isna (not `if std`/np.isnan) -- a degenerate group (all-NaN,
        # or a single name) can produce pandas' nullable pd.NA rather
        # than a plain float nan, and bool(pd.NA) raises rather than
        # returning False, which crashed on real data (a single-name or
        # all-missing (date, sector) cell) despite passing every
        # synthetic test.
        if pd.isna(std) or std == 0:
            return x * 0.0
        return (x - x.mean()) / std

    for col in feature_cols:
        full_z = monthly.groupby("date")[col].transform(zscore)
        if sector_col is None:
            monthly[f"{col}_z"] = full_z
            continue

        sector_z = monthly.groupby(["date", sector_col])[col].transform(zscore)
        n = monthly.groupby(["date", sector_col])[col].transform("count")
        monthly[f"{col}_z"] = np.where(n >= min_group_size, sector_z, full_z)

    return monthly


def build_feature_panel(
    daily_panel: pd.DataFrame,
    membership: pd.DataFrame,
    ibes_estimates: pd.DataFrame,
    ibes_actuals: pd.DataFrame,
    short_interest: pd.DataFrame,
    sector_crosswalk: pd.DataFrame,
    siccd_history: pd.DataFrame,
    horizon: int = 1,
    min_sector_group_size: int = 10,
) -> pd.DataFrame:
    """
    Orchestrates the full feature pipeline. Entry point for main.py.

    `horizon` controls how many months forward `fwd_ret` looks -- see
    `add_forward_return_target`. Changing it from the 1-month default
    changes what the model predicts (e.g. horizon=3 -> next-quarter
    relative return); model.py's walk-forward split must be given the
    same horizon so it can embargo training labels that would otherwise
    overlap the test period (see model.walk_forward_splits docstring).

    The 5 fundamental-source dataframes are pre-loaded by the caller
    (same convention as `daily_panel`/`membership` already were) --
    see load_wrds_fundamentals.py for how they're built and cached.
    """
    monthly = resample_to_monthly(daily_panel)
    monthly = add_momentum_features(monthly)
    monthly = add_volatility_feature(monthly)
    monthly = add_size_and_liquidity_features(monthly)

    # Fundamentals computed BEFORE filter_to_membership, same as the
    # price features above -- see filter_to_membership's docstring on
    # why (trailing-window features, like SUE's 8-quarter surprise
    # history, would otherwise be starved of legitimate pre-membership
    # data). Each of these enforces its own point-in-time discipline
    # internally via pit_merge.merge_asof_pit -- see their docstrings.
    monthly = add_analyst_revision_features(monthly, ibes_estimates)
    monthly = add_sue_feature(monthly, ibes_actuals, ibes_estimates)
    monthly = add_short_interest_feature(monthly, short_interest)

    monthly = add_forward_return_target(monthly, horizon=horizon)

    # Survivorship-bias fix applied here -- see filter_to_membership docstring
    # for why this happens at this specific point in the pipeline.
    monthly = filter_to_membership(monthly, membership)

    # Sector has no lookback window (unlike the fundamentals above), so
    # attaching it after the membership filter is both correct and
    # cheaper -- see attach_sector's docstring.
    monthly = attach_sector(monthly, sector_crosswalk, siccd_history)

    price_cols = ["mom_1m", "mom_3m", "mom_12m_ex1", "realized_vol", "log_mkt_cap", "log_dollar_vol"]
    fundamental_cols = ["est_rev_3m", "rev_breadth", "sue", "short_ratio"]
    feature_cols = price_cols + fundamental_cols
    monthly = cross_sectional_normalize(
        monthly, feature_cols, sector_col="sector", min_group_size=min_sector_group_size
    )

    # Coverage-gated missingness indicators -- checked, not assumed.
    # Computed on the RAW (pre-fill) coverage, since a 0-filled z-score
    # is indistinguishable from "genuinely average" otherwise. Stored on
    # .attrs (same convention as model.summarize_ic's t_stat/p_value)
    # rather than silently added to FEATURE_COLS -- the instructions give
    # an exact, closed 8-column model input list with no room for them.
    coverage = monthly[feature_cols].notna().mean()
    low_coverage_cols = coverage[coverage < 0.80].index.tolist()
    for col in low_coverage_cols:
        monthly[f"{col}_missing"] = monthly[col].isna().astype(int)
    # .to_dict() (plain Python floats), not the Series itself -- pandas
    # tries to JSON-serialize .attrs on to_parquet, and a bare Series
    # isn't JSON-serializable (caught by an actual to_parquet call on
    # real data; every earlier test only inspected .attrs directly and
    # never round-tripped through parquet).
    monthly.attrs["coverage"] = coverage.astype(float).to_dict()
    monthly.attrs["low_coverage_cols"] = low_coverage_cols

    # Fundamentals have real coverage gaps (IBES gaps, new entrants
    # without surprise history) -- do NOT dropna on them, that would
    # silently shrink and bias the cross-section toward well-covered
    # mega-caps. Instead: NaNs were already excluded from each z-score's
    # mean/std (see cross_sectional_normalize/zscore), so a NaN feature
    # here just means a NaN z-score -- fill those to 0 (cross-sectionally
    # neutral) rather than drop the row. Price features deliberately do
    # NOT get this treatment (see `required` below) -- a missing price
    # z-score is a genuine data problem (e.g. insufficient trailing
    # history), not expected fundamentals coverage, so those rows should
    # still be dropped, not silently kept as a zero.
    fundamental_z_cols = [f"{c}_z" for c in fundamental_cols]
    monthly[fundamental_z_cols] = monthly[fundamental_z_cols].fillna(0.0)

    # Only require non-null for fwd_ret and the 4 price-derived
    # z-features actually fed to the model (see model.FEATURE_COLS) --
    # NOT mom_1m_z/log_dollar_vol_z (computed but not fed to the model)
    # and NOT the fundamentals (that's the whole point of the fill-to-0
    # policy above). This must run AFTER the fundamental fillna above,
    # not before/combined with it -- these 4 columns are intentionally
    # excluded from that fillna so a genuinely missing price feature
    # still drops the row instead of silently becoming a kept zero.
    required = ["mom_3m_z", "mom_12m_ex1_z", "realized_vol_z", "log_mkt_cap_z", "fwd_ret"]
    monthly = monthly.dropna(subset=required)

    return monthly
