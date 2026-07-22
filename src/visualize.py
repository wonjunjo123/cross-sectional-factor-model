"""
visualize.py

Turns output/model_comparison.csv into a single at-a-glance figure. The
comparison table's real story is gross-vs-net (the cost impact) and whether
the Sharpe edge clears statistical significance -- so each panel is a
dumbbell (gross -> net) per model rather than a plain bar chart, with the
Sharpe panel additionally showing its bootstrap 95% CI and p-value.

Run standalone:

    python src/visualize.py

or import plot_model_comparison() and call it right after
compare_models(...).to_csv(...) in main.py.
"""

import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "output"

# Categorical slots, fixed order (matches the order models appear in the
# comparison table): slot 1 blue = linear, slot 2 orange = gbm.
COLORS = {"linear": "#2a78d6", "gbm": "#eb6834"}
MODEL_LABELS = {"linear": "Linear (Fama-MacBeth)", "gbm": "LightGBM"}

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


def _dumbbell(ax, df, gross_col, net_col, title, fmt="{:.2f}",
              gross_ci=None, net_ci=None):
    """One row per model: a thin line from gross (light) to net (full
    color), i.e. the standard 'before -> after' dumbbell for a cost-impact
    comparison. Optional CI whiskers for gross/net (used for Sharpe only,
    since that's the only metric with a bootstrap CI in the table)."""
    models = list(df.index)

    for i, model in enumerate(models):
        color = COLORS.get(model, INK_MUTED)
        gross, net = df.loc[model, gross_col], df.loc[model, net_col]

        if gross_ci:
            lo, hi = df.loc[model, gross_ci[0]], df.loc[model, gross_ci[1]]
            ax.plot([lo, hi], [i - 0.14, i - 0.14], color=color, alpha=0.3,
                     linewidth=1.5, solid_capstyle="round", zorder=0)
        if net_ci:
            lo, hi = df.loc[model, net_ci[0]], df.loc[model, net_ci[1]]
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
    ax.set_yticks(range(len(models)))
    ax.set_yticklabels([MODEL_LABELS.get(m, m) for m in models], color=INK)
    ax.set_ylim(-0.7, len(models) - 0.3)
    ax.set_title(title, color=INK, fontsize=11, fontweight="bold", loc="left")
    _style_axes(ax)


def _turnover_panel(ax, df):
    models = list(df.index)
    colors = [COLORS.get(m, INK_MUTED) for m in models]
    turnover = df["turnover_mean"]

    ax.barh(range(len(models)), turnover, color=colors, height=0.5, zorder=2)
    for i, m in enumerate(models):
        ax.annotate(f"{turnover[m]:.2f}x", (turnover[m], i),
                    textcoords="offset points", xytext=(6, 0), va="center",
                    fontsize=8.5, color=INK)

    ax.set_yticks(range(len(models)))
    ax.set_yticklabels([MODEL_LABELS.get(m, m) for m in models], color=INK)
    ax.set_ylim(-0.7, len(models) - 0.3)
    ax.set_xlim(0, max(turnover) * 1.25)
    ax.set_title("Avg. quarterly turnover (long + short legs)", color=INK,
                 fontsize=11, fontweight="bold", loc="left")
    _style_axes(ax)


def plot_model_comparison(csv_path=None, out_path=None):
    csv_path = Path(csv_path) if csv_path else OUTPUT_DIR / "model_comparison.csv"
    out_path = Path(out_path) if out_path else OUTPUT_DIR / "model_comparison.png"

    df = pd.read_csv(csv_path, index_col="model")

    fig, axes = plt.subplots(2, 2, figsize=(11, 8), facecolor=SURFACE)
    fig.suptitle("Model comparison: gross vs. net of transaction costs",
                 fontsize=13, fontweight="bold", color=INK, x=0.02, ha="left")
    fig.text(
        0.02, 0.945,
        f"Long top-decile / short bottom-decile, {int(df['n_periods'].iloc[0])} "
        f"non-overlapping quarters, {df['cost_bps'].iloc[0]:.0f}bps assumed "
        "round-trip cost -- light marker = gross, dark marker = net",
        fontsize=9, color=INK_SECONDARY,
    )

    _dumbbell(axes[0, 0], df, "sharpe", "sharpe_net", "Sharpe ratio",
              gross_ci=("sharpe_ci_low", "sharpe_ci_high"),
              net_ci=("sharpe_net_ci_low", "sharpe_net_ci_high"))
    _dumbbell(axes[0, 1], df, "ann_return", "ann_return_net",
              "Annualized return", fmt="{:+.1%}")
    _dumbbell(axes[1, 0], df, "max_drawdown", "max_drawdown_net",
              "Max drawdown", fmt="{:.1%}")
    _turnover_panel(axes[1, 1], df)

    p_lines = [
        f"{MODEL_LABELS.get(m, m)}: net Sharpe p={df.loc[m, 'sharpe_net_p_value']:.2f}"
        for m in df.index
    ]
    fig.text(0.02, 0.005,
              "Bootstrap p-value, H0: Sharpe = 0 (net) -- " + "   |   ".join(p_lines),
              fontsize=8.5, color=INK_MUTED)

    fig.tight_layout(rect=(0, 0.03, 1, 0.92))
    fig.savefig(out_path, dpi=150, facecolor=SURFACE)
    plt.close(fig)
    print(f"Saved comparison chart to {out_path}")
    return out_path


if __name__ == "__main__":
    plot_model_comparison()