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

import pandas as pd
import numpy as np


def resample_to_monthly(panel: pd.DataFrame) -> pd.DataFrame:
    """
    Collapses the daily CRSP panel to month-end observations per PERMNO.

    Builds a cumulative return index per PERMNO from CRSP's `ret` (total
    return, dividend- and delisting-adjusted by CRSP itself) rather than
    from a price series -- this is the correct way to compute momentum on
    CRSP data, since `ret` already embeds all the adjustments that the
    earlier yfinance version had to approximate via adjusted close.
    """
    panel = panel.sort_values(["permno", "date"]).copy()
    panel["cum_ret_index"] = panel.groupby("permno")["ret"].apply(
        lambda r: (1 + r.fillna(0)).cumprod()
    ).reset_index(level=0, drop=True)

    panel["month"] = panel["date"].dt.to_period("M")

    monthly = (
        panel.groupby(["permno", "month"])
        .agg(
            cum_ret_index=("cum_ret_index", "last"),
            mkt_cap=("mkt_cap", "last"),
            avg_dollar_vol=("dollar_vol", "mean"),
            daily_ret_std=("ret", "std"),  # trailing realized vol, within-month
        )
        .reset_index()
    )
    monthly["date"] = monthly["month"].dt.to_timestamp("M")
    return monthly.drop(columns="month").sort_values(["permno", "date"])


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
    monthly["mom_12m_ex1"] = g.shift(1) / g.shift(12) - 1
    return monthly


def add_volatility_feature(monthly: pd.DataFrame) -> pd.DataFrame:
    """Trailing realized daily-return volatility, already aggregated in resample step."""
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


def add_forward_return_target(monthly: pd.DataFrame, horizon: int = 1) -> pd.DataFrame:
    """
    Target variable: forward return over `horizon` months, computed PER
    PERMNO with a negative shift (pulls the FUTURE value back to the
    current row). This is what the model predicts.
    """
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
    that month -- and OLS is especially sensitive to a few extreme points.
    """
    lo, hi = x.quantile(lower), x.quantile(upper)
    return x.clip(lower=lo, upper=hi)


def cross_sectional_normalize(monthly: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
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
    """
    monthly = monthly.copy()

    def zscore(x):
        x = winsorize(x)
        return (x - x.mean()) / x.std()

    for col in feature_cols:
        monthly[f"{col}_z"] = monthly.groupby("date")[col].transform(zscore)

    return monthly


def build_feature_panel(daily_panel: pd.DataFrame, membership: pd.DataFrame) -> pd.DataFrame:
    """Orchestrates the full feature pipeline. Entry point for main.py."""
    monthly = resample_to_monthly(daily_panel)
    monthly = add_momentum_features(monthly)
    monthly = add_volatility_feature(monthly)
    monthly = add_size_and_liquidity_features(monthly)
    monthly = add_forward_return_target(monthly, horizon=1)

    # Survivorship-bias fix applied here -- see filter_to_membership docstring
    # for why this happens at this specific point in the pipeline.
    monthly = filter_to_membership(monthly, membership)

    feature_cols = [
        "mom_1m", "mom_3m", "mom_12m_ex1", "realized_vol",
        "log_mkt_cap", "log_dollar_vol",
    ]
    monthly = cross_sectional_normalize(monthly, feature_cols)

    z_cols = [f"{c}_z" for c in feature_cols]
    monthly = monthly.dropna(subset=z_cols + ["fwd_ret"])

    return monthly
