"""
backtest.py

Turns model predictions into a long-short decile portfolio and computes
standard performance metrics. This is the module where you produce the
"headline" numbers (Sharpe, drawdown, turnover) -- but the honest version
of this project treats turnover as a first-class output, not a footnote,
since it's what determines whether the strategy survives contact with
real transaction costs.
"""

import pandas as pd
import numpy as np


def assign_deciles(predictions: pd.DataFrame) -> pd.DataFrame:
    """
    Ranks stocks into deciles by predicted score, WITHIN each date.

    Ties in `pred` (e.g. multiple stocks routed to the same GBM leaf) are
    broken by `rank(method="first")` before qcut, so qcut's quantile edges
    are computed on a column of guaranteed-unique per-date ranks and can
    never collide. Without this, a single tie could make qcut's
    `duplicates="drop"` silently produce FEWER than 10 bins for that date,
    which shifts every label above the merge point down by one -- so the
    top decile would no longer be labeled 9, and every caller that checks
    `decile == 9` (the long leg) would silently see zero rows for that
    date instead of the real top group. `duplicates="drop"` is kept only
    as a defensive fallback for a cross-section genuinely smaller than 10
    names, which shouldn't happen on this S&P 500 universe.
    """
    predictions = predictions.copy()
    predictions["decile"] = predictions.groupby("date")["pred"].transform(
        lambda x: pd.qcut(x.rank(method="first"), 10, labels=False, duplicates="drop")
    )
    return predictions


def compute_portfolio_returns(predictions: pd.DataFrame) -> pd.DataFrame:
    """
    Long top decile (9), short bottom decile (0), equal-weighted within
    each leg. Returns a monthly return series for the long-short portfolio,
    plus the long-only and short-only legs separately (useful for
    diagnosing whether the edge is coming from the long side, the short
    side, or both -- a common follow-up question).
    """
    predictions = assign_deciles(predictions)

    long_leg = (
        predictions[predictions["decile"] == 9]
        .groupby("date")["fwd_ret"].mean()
        .rename("long_ret")
    )
    short_leg = (
        predictions[predictions["decile"] == 0]
        .groupby("date")["fwd_ret"].mean()
        .rename("short_ret")
    )

    port = pd.concat([long_leg, short_leg], axis=1).dropna()
    port["ls_ret"] = port["long_ret"] - port["short_ret"]
    return port.reset_index()


def compute_turnover(predictions: pd.DataFrame, decile: int = 9) -> pd.Series:
    """
    Period-over-period fraction of names in `decile` that changed.
    decile=9 is the long leg, decile=0 is the short leg -- both are needed
    to cost the full long-short book (see apply_transaction_costs), not
    just the long side. High turnover erodes an otherwise-real edge once
    you account for transaction costs -- this number is what tells you
    whether the strategy is realistic or a backtest artifact.
    """
    top_decile = assign_deciles(predictions)
    top_decile = top_decile[top_decile["decile"] == decile]

    holdings_by_date = top_decile.groupby("date")["permno"].apply(set)
    dates = sorted(holdings_by_date.index)

    turnover = []
    for i in range(1, len(dates)):
        prev_set = holdings_by_date[dates[i - 1]]
        curr_set = holdings_by_date[dates[i]]
        changed = len(curr_set - prev_set)
        turnover.append(changed / max(len(curr_set), 1))

    return pd.Series(turnover, index=dates[1:], name="turnover")


def apply_transaction_costs(
    port: pd.DataFrame, predictions: pd.DataFrame, cost_bps: float = 20.0
) -> pd.DataFrame:
    """
    Deducts an assumed round-trip transaction cost from each period's
    long-short return, sized to the ACTUAL long-leg + short-leg turnover
    that period -- not a flat haircut. Replacing 75% of a decile's names
    costs far more than replacing 20%, and turnover is exactly the input
    that determines how much of a backtested return survives contact with
    real trading costs.

    `cost_bps`: assumed round-trip cost (bid-ask spread + market impact +
    commissions combined) in basis points, charged once per unit of
    turnover on EACH leg. 20bps is a deliberately simple, conservative
    mid-point for liquid large-cap S&P 500 names -- real costs vary by
    name, period, and order size; this is a back-of-envelope estimate, not
    a microstructure/impact model.

    The very first period has no prior holdings to diff against, so
    turnover can't be measured there -- it's treated as 100% turnover on
    both legs (a full initial buy-in from cash), not 0%, since assuming a
    free first trade would understate costs.
    """
    long_turnover = compute_turnover(predictions, decile=9)
    short_turnover = compute_turnover(predictions, decile=0)

    port = port.set_index("date")
    long_turnover = long_turnover.reindex(port.index).fillna(1.0)
    short_turnover = short_turnover.reindex(port.index).fillna(1.0)
    combined_turnover = long_turnover + short_turnover

    port = port.copy()
    port["turnover_combined"] = combined_turnover
    port["cost_drag"] = combined_turnover * (cost_bps / 10000)
    port["ls_ret_net"] = port["ls_ret"] - port["cost_drag"]

    # Long-only variant (see compare_portfolio_constructions): no short leg
    # to trade, so only the long leg's own turnover is chargeable.
    port["long_ret_net"] = port["long_ret"] - long_turnover * (cost_bps / 10000)

    return port.reset_index()


def bootstrap_sharpe_test(
    returns: pd.Series,
    freq: float = 12,
    n_bootstrap: int = 10000,
    seed: int = 0,
) -> dict:
    """
    Bootstrap significance test for a Sharpe ratio estimated on a small
    sample. An i.i.d. resampling bootstrap like this is only valid because
    `returns` is a NON-OVERLAPPING return series (one observation per
    rebalance period -- see model.run_walk_forward's step_months=HORIZON
    reasoning); it would be invalid on the overlapping/autocorrelated
    return series that non-overlapping evaluation was specifically built to
    avoid.

    Reports two distinct things, since they answer different questions:
      - `ci_low`/`ci_high`: a 95% percentile bootstrap CI for the Sharpe,
        from resampling the observed returns as-is.
      - `p_value`: a two-sided p-value for H0: true Sharpe == 0, from a
        NULL-CENTERED bootstrap -- returns are shifted to exactly zero mean
        (imposing H0) before resampling, then p is the fraction of
        null-world resamples whose |Sharpe| is at least as extreme as the
        one actually observed. This is NOT the same as checking whether 0
        falls in the CI above; both are reported because a small sample can
        show a wide CI while still failing to cleanly reject H0.
    """
    r = returns.dropna().to_numpy()
    n = len(r)
    rng = np.random.default_rng(seed)

    def sharpe(samples: np.ndarray) -> np.ndarray:
        means = samples.mean(axis=1)
        stds = samples.std(axis=1, ddof=1)
        return np.where(stds > 0, (means / stds) * np.sqrt(freq), np.nan)

    observed = sharpe(r.reshape(1, -1))[0]

    boot_ci = sharpe(rng.choice(r, size=(n_bootstrap, n), replace=True))
    ci_low, ci_high = np.nanpercentile(boot_ci, [2.5, 97.5])

    r_null = r - r.mean()
    boot_null = sharpe(rng.choice(r_null, size=(n_bootstrap, n), replace=True))
    boot_null = boot_null[~np.isnan(boot_null)]
    p_value = float(np.mean(np.abs(boot_null) >= abs(observed)))

    return {
        "sharpe_ci_low": ci_low,
        "sharpe_ci_high": ci_high,
        "sharpe_p_value": p_value,
        "n_bootstrap": n_bootstrap,
    }


def performance_summary(port: pd.DataFrame, freq: float = 12, ret_col: str = "ls_ret") -> dict:
    """
    Standard annualized performance metrics from a return series.
    freq=12 annualizes monthly returns; adjust if you change rebalance
    frequency. ret_col defaults to the gross long-short return
    ("ls_ret") -- pass ret_col="ls_ret_net" (see apply_transaction_costs)
    to get the same metrics net of assumed transaction costs.
    """
    ret = port[ret_col]
    ann_return = ret.mean() * freq
    ann_vol = ret.std() * np.sqrt(freq)
    sharpe = ann_return / ann_vol if ann_vol > 0 else np.nan

    cum = (1 + ret).cumprod()
    running_max = cum.cummax()
    drawdown = (cum - running_max) / running_max
    max_dd = drawdown.min()

    return {
        "ann_return": ann_return,
        "ann_vol": ann_vol,
        "sharpe": sharpe,
        "max_drawdown": max_dd,
        "n_periods": len(ret),
    }


_COMPARISON_COLS = [
    "ann_return", "ann_vol", "sharpe", "sharpe_ci_low", "sharpe_ci_high",
    "sharpe_p_value", "max_drawdown", "ann_return_net", "sharpe_net",
    "sharpe_net_ci_low", "sharpe_net_ci_high", "sharpe_net_p_value",
    "max_drawdown_net", "turnover_mean", "cost_bps", "n_periods",
]


def _return_stats(port: pd.DataFrame, ret_col: str, freq: float) -> dict:
    """Annualized performance + bootstrap significance for one return
    column. Shared by compare_models (gross/net long-short) and
    compare_portfolio_constructions (long-short vs. long-only)."""
    perf = performance_summary(port, freq=freq, ret_col=ret_col)
    perf.update(bootstrap_sharpe_test(port[ret_col], freq=freq))
    return perf


def _gross_net_row(port: pd.DataFrame, gross_col: str, net_col: str, freq: float) -> dict:
    """One comparison-table row: gross stats plus the net_col variants
    merged in under the `_net` suffix -- the shape both compare_models
    and compare_portfolio_constructions build their rows from."""
    perf = _return_stats(port, gross_col, freq)

    net = _return_stats(port, net_col, freq)
    perf["ann_return_net"] = net["ann_return"]
    perf["sharpe_net"] = net["sharpe"]
    perf["max_drawdown_net"] = net["max_drawdown"]
    perf["sharpe_net_ci_low"] = net["sharpe_ci_low"]
    perf["sharpe_net_ci_high"] = net["sharpe_ci_high"]
    perf["sharpe_net_p_value"] = net["sharpe_p_value"]
    return perf


def compare_models(
    predictions_by_model: dict[str, pd.DataFrame], freq: float = 12, cost_bps: float = 20.0
) -> pd.DataFrame:
    """
    Runs the full backtest for each model's predictions and returns a
    performance table -- this is the table your write-up's "findings"
    section is built around: does the model have a real edge, net of
    realistic trading costs, and is that edge even statistically
    distinguishable from zero?

    `freq` is the number of rebalance periods per year, used to annualize
    return/vol -- 12 for monthly rebalancing, 4 for quarterly, etc. Must
    match the `test_months`/`step_months` actually used to produce these
    predictions in model.run_walk_forward, or the annualized numbers below
    are simply wrong. `cost_bps` is passed straight to
    apply_transaction_costs (see its docstring for what it represents).

    Reports both gross and net-of-cost figures side by side, each with its
    own bootstrap significance check -- costs and significance are
    evaluated independently because a gross edge that's already
    insignificant can only get worse net of costs, and collapsing them
    into one number would hide which of the two is doing the damage.
    """
    rows = []
    for name, preds in predictions_by_model.items():
        port = compute_portfolio_returns(preds)
        port = apply_transaction_costs(port, preds, cost_bps=cost_bps)

        perf = _gross_net_row(port, "ls_ret", "ls_ret_net", freq)
        perf["turnover_mean"] = compute_turnover(preds).mean()
        perf["cost_bps"] = cost_bps
        perf["model"] = name
        rows.append(perf)

    return pd.DataFrame(rows).set_index("model")[_COMPARISON_COLS]


def compare_portfolio_constructions(
    predictions: pd.DataFrame, freq: float = 12, cost_bps: float = 20.0
) -> pd.DataFrame:
    """
    Compares the full long-short book (long top decile, short bottom
    decile) against a long-only variant (top decile only, no short leg)
    built from the SAME model predictions -- answers "how much of this
    edge, or this cost drag, is actually coming from the short leg?"
    rather than comparing across different models.

    `turnover_mean` means the turnover base each row's costs are actually
    charged against: combined long+short turnover for the `long_short`
    row, long-leg-only turnover for `long_only` (there's no short leg to
    turn over there). No short-selling costs beyond that turnover-based
    charge -- no borrow fee/margin modeling -- are added for either row;
    see apply_transaction_costs and the README's known limitations.

    Returns a table shaped like `compare_models`'s output, but indexed by
    `construction` ("long_short" / "long_only") instead of `model`.
    """
    port = compute_portfolio_returns(predictions)
    port = apply_transaction_costs(port, predictions, cost_bps=cost_bps)

    long_turnover = compute_turnover(predictions, decile=9)
    short_turnover = compute_turnover(predictions, decile=0)

    constructions = {
        "long_short": ("ls_ret", "ls_ret_net", (long_turnover + short_turnover).mean()),
        "long_only": ("long_ret", "long_ret_net", long_turnover.mean()),
    }

    rows = []
    for label, (gross_col, net_col, turnover_mean) in constructions.items():
        perf = _gross_net_row(port, gross_col, net_col, freq)
        perf["turnover_mean"] = turnover_mean
        perf["cost_bps"] = cost_bps
        perf["construction"] = label
        rows.append(perf)

    return pd.DataFrame(rows).set_index("construction")[_COMPARISON_COLS]
