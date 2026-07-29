"""
tune.py

Lightweight hyperparameter-sweep harness for model.run_walk_forward. Not
part of the production pipeline (main.py) -- this exists for the tuning
roadmap in hyperparameter_approach.md: given a parameter override, run
the full walk-forward with diagnostics on, score it on mean OOS IC / IC
IR / in-sample-vs-OOS gap / the per-fold IC series (not just a single
aggregate number, since label dispersion shifts across regimes -- see
the roadmap), and log the result to a CSV so sweeps are comparable side
by side across separate runs instead of scattered across notebook cells.

Sweeps are ONE parameter at a time (coordinate descent), matching the
roadmap's own tiered structure -- not a combinatorial grid across every
parameter in a tier. Lock in the best value for one parameter (via
`base_params`), then sweep the next one against it.

Usage:
    python src/tune.py                      # runs the Tier 1 sweeps
    from tune import load_feature_panel, sweep, run_experiment
"""

from __future__ import annotations

import time
from pathlib import Path

import pandas as pd

from features import build_feature_panel
from model import run_walk_forward, DEFAULT_MODEL_PARAMS

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "output"
RESULTS_PATH = OUTPUT_DIR / "tuning_results.csv"

# Must match main.py's HORIZON -- see model.run_walk_forward's docstring
# on why test_months=1, step_months=HORIZON is required together.
HORIZON = 3


def load_feature_panel() -> pd.DataFrame:
    """Builds the same feature panel main.py trains on, once, so a sweep
    of many experiments doesn't re-read/re-build it per config."""
    daily_panel = pd.read_parquet(DATA_DIR / "prices_wrds.parquet")
    membership = pd.read_parquet(DATA_DIR / "sp500_membership.parquet")
    return build_feature_panel(daily_panel, membership, horizon=HORIZON)


def run_experiment(
    feature_panel: pd.DataFrame,
    model_params: dict,
    label: str = "",
    train_months: int = 60,
    test_months: int = 1,
    step_months: int = HORIZON,
    horizon: int = HORIZON,
) -> dict:
    """
    Runs one full walk-forward pass with `model_params` overriding
    DEFAULT_MODEL_PARAMS, and returns a single summary row: mean/std/IR
    of OOS IC, mean in-sample IC, mean train-vs-OOS gap, and the
    per-fold OOS IC series itself (so the distribution is inspectable
    without re-running -- a hyperparameter that raises mean IC while
    making some quarters wildly negative is not actually an improvement,
    per the roadmap's own framing).
    """
    full_params = {**DEFAULT_MODEL_PARAMS, **model_params}

    start = time.time()
    _, diagnostics = run_walk_forward(
        feature_panel, train_months=train_months, test_months=test_months,
        step_months=step_months, horizon=horizon,
        model_params=model_params, return_diagnostics=True,
    )
    elapsed = time.time() - start

    oos_ic = diagnostics["oos_ic"]
    return {
        "label": label,
        "model_params": full_params,
        "oos_ic_mean": oos_ic.mean(),
        "oos_ic_std": oos_ic.std(),
        "oos_ic_ir": oos_ic.mean() / oos_ic.std() if oos_ic.std() > 0 else float("nan"),
        "in_sample_ic_mean": diagnostics["in_sample_ic"].mean(),
        "gap_mean": diagnostics["gap"].mean(),
        "n_folds": len(diagnostics),
        "per_fold_oos_ic": oos_ic.round(4).tolist(),
        "seconds": round(elapsed, 1),
    }


def sweep(
    feature_panel: pd.DataFrame,
    param_name: str,
    values: list,
    base_params: dict | None = None,
    save: bool = True,
    **run_walk_forward_kwargs,
) -> pd.DataFrame:
    """
    Sweeps ONE hyperparameter across `values`, holding every other
    parameter at `base_params` (default: plain DEFAULT_MODEL_PARAMS) --
    coordinate-descent style, not a combinatorial grid (see module
    docstring). Pass a `base_params` with an already-chosen value locked
    in (e.g. {"max_depth": 3}) to sweep the NEXT parameter against it.
    Appends each result to RESULTS_PATH as it runs, so a sweep survives
    being interrupted partway and stays comparable against every other
    sweep run in this or a past session.
    """
    base_params = dict(base_params or {})
    rows = []
    for value in values:
        params = {**base_params, param_name: value}
        label = f"{param_name}={value}"
        print(f"\n--- {label} ---")
        row = run_experiment(feature_panel, params, label=label, **run_walk_forward_kwargs)
        print(f"  oos_ic_mean={row['oos_ic_mean']:+.4f}  oos_ic_ir={row['oos_ic_ir']:+.3f}  "
              f"in_sample_ic={row['in_sample_ic_mean']:+.4f}  gap={row['gap_mean']:+.4f}  "
              f"({row['seconds']}s)")
        rows.append(row)

    results = pd.DataFrame(rows)
    if save:
        _append_results(results)
    return results


def _append_results(results: pd.DataFrame) -> None:
    """CSV can't hold nested dict/list cells -- stringify model_params and
    per_fold_oos_ic so the row is still self-contained and readable."""
    OUTPUT_DIR.mkdir(exist_ok=True)
    results = results.copy()
    results["model_params"] = results["model_params"].apply(str)
    results["per_fold_oos_ic"] = results["per_fold_oos_ic"].apply(str)
    header = not RESULTS_PATH.exists()
    results.to_csv(RESULTS_PATH, mode="a", header=header, index=False)
    print(f"\nAppended {len(results)} rows to {RESULTS_PATH}")


if __name__ == "__main__":
    panel = load_feature_panel()

    # Tier 1 -- capacity control, one parameter at a time against the
    # current defaults (see hyperparameter_approach.md). Each sweep is
    # independent (held against DEFAULT_MODEL_PARAMS, not the previous
    # sweep's winner) -- inspect output/tuning_results.csv and pick
    # winners manually before chaining a base_params-locked sweep.
    sweep(panel, "max_depth", [2, 3, 4, 5, 6])
    sweep(panel, "min_child_weight", [5, 15, 30, 60])
    sweep(panel, "subsample", [0.6, 0.7, 0.8, 1.0])
    sweep(panel, "colsample_bytree", [0.6, 0.7, 0.8, 1.0])
