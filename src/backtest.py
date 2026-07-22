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
    """Ranks stocks into deciles by predicted score, WITHIN each date."""
    predictions = predictions.copy()
    predictions["decile"] = predictions.groupby("date")["pred"].transform(
        lambda x: pd.qcut(x, 10, labels=False, duplicates="drop")
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


def compute_turnover(predictions: pd.DataFrame) -> pd.Series:
    """
    Month-over-month fraction of names in the top decile that changed.
    High turnover erodes an otherwise-real edge once you account for
    transaction costs -- this number is what tells you whether the
    strategy is realistic or a backtest artifact.
    """
    top_decile = assign_deciles(predictions)
    top_decile = top_decile[top_decile["decile"] == 9]

    holdings_by_date = top_decile.groupby("date")["permno"].apply(set)
    dates = sorted(holdings_by_date.index)

    turnover = []
    for i in range(1, len(dates)):
        prev_set = holdings_by_date[dates[i - 1]]
        curr_set = holdings_by_date[dates[i]]
        changed = len(curr_set - prev_set)
        turnover.append(changed / max(len(curr_set), 1))

    return pd.Series(turnover, index=dates[1:], name="turnover")


def performance_summary(port: pd.DataFrame, freq: int = 12) -> dict:
    """
    Standard annualized performance metrics from a monthly return series.
    freq=12 annualizes monthly returns; adjust if you change rebalance
    frequency.
    """
    ret = port["ls_ret"]
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


def compare_models(predictions_by_model: dict[str, pd.DataFrame], freq: int = 12) -> pd.DataFrame:
    """
    Runs the full backtest for each model's predictions and returns a
    side-by-side comparison table -- this is the table your write-up's
    "findings" section is built around: does the ML model actually beat
    the linear baseline, and at what turnover cost?

    `freq` is the number of rebalance periods per year, used to annualize
    return/vol -- 12 for monthly rebalancing, 4 for quarterly, etc. Must
    match the `test_months`/`step_months` actually used to produce these
    predictions in model.run_walk_forward, or the annualized numbers below
    are simply wrong.
    """
    rows = []
    for name, preds in predictions_by_model.items():
        port = compute_portfolio_returns(preds)
        perf = performance_summary(port, freq=freq)
        perf["turnover_mean"] = compute_turnover(preds).mean()
        perf["model"] = name
        rows.append(perf)

    return pd.DataFrame(rows).set_index("model")[
        ["ann_return", "ann_vol", "sharpe", "max_drawdown", "turnover_mean", "n_periods"]
    ]
