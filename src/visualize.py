"""
visualize.py

Turns output/portfolio_construction_comparison.csv into a single
at-a-glance figure. The comparison table's real story is gross-vs-net
(the cost impact) and whether the Sharpe edge clears statistical
significance -- so each panel is a dumbbell (gross -> net) per row
rather than a plain bar chart, with the Sharpe panel additionally
showing its bootstrap 95% CI and p-value.

Run standalone:

    python src/visualize.py

or import plot_portfolio_construction_comparison() and call it right
after compare_portfolio_constructions(...).to_csv(...) in main.py.
"""

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
from pathlib import Path

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "output"

COLORS = {"gbm": "#2a78d6"}
MODEL_LABELS = {"gbm": "XGBoost"}

# Same categorical slots (blue, orange) this project used back when there
# were two models to distinguish -- reused here for the two portfolio
# CONSTRUCTIONS instead (see backtest.compare_portfolio_constructions).
CONSTRUCTION_COLORS = {"long_short": "#2a78d6", "long_only": "#eb6834"}
CONSTRUCTION_LABELS = {"long_short": "Long-short", "long_only": "Long-only"}

INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRIDLINE = "#e1e0d9"
BASELINE = "#c3c2b7"
SURFACE = "#fcfcfb"


def _style_axes(ax):
    ax.set_facecolor(SURFACE)
    ax.tick_params(colors=INK_MUTED, length=0)
    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)
    ax.spines["bottom"].set_color(BASELINE)
    ax.xaxis.grid(True, color=GRIDLINE, linewidth=0.8, zorder=-1)
    ax.set_axisbelow(True)


def _dumbbell(ax, df, gross_col, net_col, title, colors, labels, fmt="{:.2f}",
              gross_ci=None, net_ci=None):
    """One row per index entry: a thin line from gross (light) to net
    (full color), i.e. the standard 'before -> after' dumbbell for a
    cost-impact comparison. `colors`/`labels` map each df.index value to
    a hex color / display name, so this generalizes to any row identity
    (models, portfolio constructions, ...). Optional CI whiskers for
    gross/net (used for Sharpe only, since that's the only metric with a
    bootstrap CI in the table)."""
    rows = list(df.index)

    for i, row in enumerate(rows):
        color = colors.get(row, INK_MUTED)
        gross, net = df.loc[row, gross_col], df.loc[row, net_col]

        if gross_ci:
            lo, hi = df.loc[row, gross_ci[0]], df.loc[row, gross_ci[1]]
            ax.plot([lo, hi], [i - 0.14, i - 0.14], color=color, alpha=0.3,
                     linewidth=1.5, solid_capstyle="round", zorder=0)
        if net_ci:
            lo, hi = df.loc[row, net_ci[0]], df.loc[row, net_ci[1]]
            ax.plot([lo, hi], [i + 0.14, i + 0.14], color=color, alpha=0.55,
                     linewidth=1.5, solid_capstyle="round", zorder=0)

        ax.plot([gross, net], [i, i], color=color, linewidth=1.5, zorder=1,
                 solid_capstyle="round")
        ax.scatter([gross], [i], s=70, color=color, alpha=0.4, zorder=2,
                   edgecolor=SURFACE, linewidth=1)
        ax.scatter([net], [i], s=70, color=color, alpha=1.0, zorder=3,
                   edgecolor=SURFACE, linewidth=1)

        ax.annotate(fmt.format(gross), (gross, i), textcoords="offset points",
                    xytext=(0, 9), ha="center", fontsize=8.5, color=INK_SECONDARY)
        ax.annotate(fmt.format(net), (net, i), textcoords="offset points",
                    xytext=(0, -13), ha="center", fontsize=8.5, color=INK,
                    fontweight="bold")

    ax.axvline(0, color=BASELINE, linewidth=1, zorder=0)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([labels.get(r, r) for r in rows], color=INK)
    ax.set_ylim(-0.7, len(rows) - 0.3)
    ax.set_title(title, color=INK, fontsize=11, fontweight="bold", loc="left")
    _style_axes(ax)


def _turnover_panel(ax, df, colors, labels):
    rows = list(df.index)
    row_colors = [colors.get(r, INK_MUTED) for r in rows]
    turnover = df["turnover_mean"]

    ax.barh(range(len(rows)), turnover, color=row_colors, height=0.5, zorder=2)
    for i, r in enumerate(rows):
        ax.annotate(f"{turnover[r]:.2f}x", (turnover[r], i),
                    textcoords="offset points", xytext=(6, 0), va="center",
                    fontsize=8.5, color=INK)

    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([labels.get(r, r) for r in rows], color=INK)
    ax.set_ylim(-0.7, len(rows) - 0.3)
    ax.set_xlim(0, max(turnover) * 1.25)
    ax.set_title("Avg. quarterly turnover", color=INK,
                 fontsize=11, fontweight="bold", loc="left")
    _style_axes(ax)


def plot_portfolio_construction_comparison(csv_path=None, out_path=None):
    """
    Reads backtest.compare_portfolio_constructions's output (long-short
    vs. long-only) and renders gross-vs-net Sharpe/return/drawdown
    dumbbells plus a turnover panel -- the summary-stat companion to
    plot_equity_curve's time-series view of the same two constructions.
    """
    csv_path = Path(csv_path) if csv_path else OUTPUT_DIR / "portfolio_construction_comparison.csv"
    out_path = Path(out_path) if out_path else OUTPUT_DIR / "portfolio_construction_comparison.png"

    df = pd.read_csv(csv_path, index_col="construction")

    fig, axes = plt.subplots(2, 2, figsize=(11, 8), facecolor=SURFACE)
    fig.suptitle("Backtest performance: long-short vs. long-only, gross vs. net",
                 fontsize=13, fontweight="bold", color=INK, x=0.02, ha="left")
    fig.text(
        0.02, 0.945,
        f"{int(df['n_periods'].iloc[0])} non-overlapping quarters, "
        f"{df['cost_bps'].iloc[0]:.0f}bps assumed round-trip cost -- "
        "light marker = gross, dark marker = net",
        fontsize=9, color=INK_SECONDARY,
    )

    _dumbbell(axes[0, 0], df, "sharpe", "sharpe_net", "Sharpe ratio",
              CONSTRUCTION_COLORS, CONSTRUCTION_LABELS,
              gross_ci=("sharpe_ci_low", "sharpe_ci_high"),
              net_ci=("sharpe_net_ci_low", "sharpe_net_ci_high"))
    _dumbbell(axes[0, 1], df, "ann_return", "ann_return_net",
              "Annualized return", CONSTRUCTION_COLORS, CONSTRUCTION_LABELS,
              fmt="{:+.1%}")
    _dumbbell(axes[1, 0], df, "max_drawdown", "max_drawdown_net",
              "Max drawdown", CONSTRUCTION_COLORS, CONSTRUCTION_LABELS,
              fmt="{:.1%}")
    _turnover_panel(axes[1, 1], df, CONSTRUCTION_COLORS, CONSTRUCTION_LABELS)

    p_lines = [
        f"{CONSTRUCTION_LABELS.get(c, c)}: net Sharpe p={df.loc[c, 'sharpe_net_p_value']:.2f}"
        for c in df.index
    ]
    fig.text(0.02, 0.005,
              "Bootstrap p-value, H0: Sharpe = 0 (net) -- " + "   |   ".join(p_lines),
              fontsize=8.5, color=INK_MUTED)

    fig.tight_layout(rect=(0, 0.03, 1, 0.92))
    fig.savefig(out_path, dpi=150, facecolor=SURFACE)
    plt.close(fig)
    print(f"Saved comparison chart to {out_path}")
    return out_path


def plot_ic_timeseries(ic_by_model, out_path=None):
    """ic_by_model: dict of {model_name: DataFrame[date, ic]} (as returned by
    model.summarize_ic) or {model_name: path-to-csv}. Drawn as a line vs a
    zero baseline -- IC's natural reference point -- since the question is
    whether it sits persistently on one side of zero, not just its magnitude.
    A dashed line at the mean IC shows that at a glance."""
    out_path = Path(out_path) if out_path else OUTPUT_DIR / "ic_timeseries.png"

    series = {}
    for model, data in ic_by_model.items():
        df = data.copy() if isinstance(data, pd.DataFrame) else pd.read_csv(data)
        df["date"] = pd.to_datetime(df["date"])
        series[model] = df.sort_values("date")

    fig, ax = plt.subplots(figsize=(10, 4.5), facecolor=SURFACE)

    # Single series: no legend needed, the title carries the mean directly
    # (dataviz convention -- one series' identity is already named by the
    # title). 2+ series: each needs its own labeled line, so a legend
    # carries the means instead -- cramming N means into the title
    # wouldn't scale and silently showing only one would be misleading.
    multi = len(series) > 1
    mean_ics = {}
    for model, df in series.items():
        color = COLORS.get(model, INK_MUTED)
        mean_ic = df["ic"].mean()
        mean_ics[model] = mean_ic
        label = f"{MODEL_LABELS.get(model, model)} (mean {mean_ic:+.3f})" if multi else None
        ax.plot(df["date"], df["ic"], color=color, linewidth=1.5, marker="o",
                markersize=5, markeredgecolor=SURFACE, markeredgewidth=0.8,
                zorder=2, label=label)
        ax.axhline(mean_ic, color=color, linewidth=1, alpha=0.35, zorder=1,
                   linestyle="--")

    ax.axhline(0, color=BASELINE, linewidth=1, zorder=0)

    title = "Out-of-sample Information Coefficient (Spearman), by quarter"
    if not multi:
        title += f" -- mean {next(iter(mean_ics.values())):+.3f}"
    ax.set_title(title, color=INK, fontsize=12, fontweight="bold", loc="left")
    ax.set_ylabel("IC", color=INK_SECONDARY, fontsize=9.5)

    if multi:
        legend = ax.legend(loc="upper right", frameon=False, fontsize=9)
        for text in legend.get_texts():
            text.set_color(INK)

    _style_axes(ax)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, facecolor=SURFACE)
    plt.close(fig)
    print(f"Saved IC time series chart to {out_path}")
    return out_path


def plot_equity_curve(returns_csv=None, out_path=None):
    """
    Cumulative growth of $1 over time, long-short vs. long-only, gross vs.
    net of costs, with net-of-cost underwater drawdown directly below.
    This is the one thing the summary figures
    (plot_portfolio_construction_comparison's bars, plot_ic_timeseries)
    don't show: HOW the return accrued --
    steadily, in one lucky quarter, wiped out and rebuilt -- not just the
    endpoint annualized stats. `returns_csv` is the per-period return
    series (date, long_ret, short_ret, ls_ret, ls_ret_net, long_ret_net,
    ...) that backtest.apply_transaction_costs produces -- already has
    both constructions' columns, so no separate file is needed for each;
    see backtest.compare_portfolio_constructions for the matching
    summary-stat table this chart is the time-series companion to.
    """
    returns_csv = Path(returns_csv) if returns_csv else OUTPUT_DIR / "portfolio_returns.csv"
    out_path = Path(out_path) if out_path else OUTPUT_DIR / "equity_curve.png"

    df = pd.read_csv(returns_csv, parse_dates=["date"]).sort_values("date")

    constructions = {
        "long_short": ("ls_ret", "ls_ret_net"),
        "long_only": ("long_ret", "long_ret_net"),
    }

    def drawdown(cum):
        return cum / cum.cummax() - 1

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(10, 7.5), facecolor=SURFACE, sharex=True,
        gridspec_kw={"height_ratios": [2.2, 1]},
    )

    for construction, (gross_col, net_col) in constructions.items():
        color = CONSTRUCTION_COLORS[construction]
        cum_gross = (1 + df[gross_col]).cumprod()
        cum_net = (1 + df[net_col]).cumprod()

        ax1.plot(df["date"], cum_gross, color=color, alpha=0.45, linewidth=1.5, zorder=1)
        ax1.plot(df["date"], cum_net, color=color, alpha=1.0, linewidth=1.5,
                  zorder=2, label=CONSTRUCTION_LABELS[construction])

        dd_net = drawdown(cum_net)
        ax2.plot(df["date"], dd_net, color=color, alpha=1.0, linewidth=1.3, zorder=2)
        ax2.fill_between(df["date"], dd_net, 0, color=color, alpha=0.15, zorder=1)

    ax1.axhline(1.0, color=BASELINE, linewidth=1, zorder=0)
    fig.suptitle("Growth of $1: long-short vs. long-only", fontsize=13,
                 fontweight="bold", color=INK, x=0.02, ha="left")
    fig.text(0.02, 0.945, "Light line = gross, dark line = net of costs",
             fontsize=9, color=INK_SECONDARY)
    ax1.set_ylabel("Growth of $1", color=INK_SECONDARY, fontsize=9.5)
    legend = ax1.legend(loc="upper left", frameon=False, fontsize=9)
    for text in legend.get_texts():
        text.set_color(INK)
    _style_axes(ax1)

    ax2.axhline(0, color=BASELINE, linewidth=1, zorder=0)
    ax2.set_title("Drawdown, net of costs", color=INK, fontsize=10.5,
                  fontweight="bold", loc="left")
    ax2.set_ylabel("Drawdown", color=INK_SECONDARY, fontsize=9.5)
    ax2.yaxis.set_major_formatter(PercentFormatter(xmax=1))
    _style_axes(ax2)

    fig.tight_layout(rect=(0, 0.02, 1, 0.90))
    fig.savefig(out_path, dpi=150, facecolor=SURFACE)
    plt.close(fig)
    print(f"Saved equity curve chart to {out_path}")
    return out_path


if __name__ == "__main__":
    plot_portfolio_construction_comparison()

    ic_path = OUTPUT_DIR / "ic_gbm.csv"
    if ic_path.exists():
        plot_ic_timeseries({"gbm": ic_path})

    returns_path = OUTPUT_DIR / "portfolio_returns.csv"
    if returns_path.exists():
        plot_equity_curve(returns_path)