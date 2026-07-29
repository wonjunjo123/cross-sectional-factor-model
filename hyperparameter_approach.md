# Hyperparameter Tuning Roadmap — XGBRanker Walk-Forward Equity Ranking Model

## Context for Claude (VSCode)

This document summarizes a hyperparameter-tuning strategy developed for a cross-sectional
equity ranking pipeline (`main.py` / `model.py` / `features.py`). The model is `XGBRanker`
with `objective="rank:ndcg"`, trained on a rolling 60-month walk-forward basis, predicting
within-month return deciles (0-9) per stock, evaluated via monthly Spearman IC (rank
correlation between predicted and realized forward returns).

**Verified data characteristics** (measured directly, not assumed):

| Characteristic | Value | Why it matters |
|---|---|---|
| Stocks per month (query group size) | ~476.6 mean, std 4.4, range 466-484 | Groups are large and stable — this is **not** a small-query-group ranking problem. Do not over-regularize `min_child_weight` on the assumption that groups are thin. |
| Total unique months in panel | 168 (~14 years) | With `train_months=60`, roughly ~35 walk-forward test windows exist at `step_months=3`. Enough to see trends, not enough to fully trust a single aggregate metric — always inspect the per-fold distribution, not just the mean. |
| Early-fold training window size | First fold ships with only 58 months, not the full 60 | Caused by `train_start = max(0, train_end - train_months)` clipping at zero combined with the embargo trim. Confirm whether this converges to a full 60-month window after a few folds. |
| Cross-sectional label dispersion (decile boundaries) | Visibly shifts month to month (e.g., early vs. mid-2012) | Expected given a 3-month horizon — quarterly return spread is regime-dependent. This is exactly why decile-ranking (not raw-return regression) is the right label design: it only needs relative order within a date, not a stable magnitude scale. But it also means IC *should* vary across time regardless of hyperparameters — a change that raises mean IC while making some quarters wildly negative is not actually an improvement. |

---

## Guiding Principle

Hyperparameters should be chosen **in an order driven by the project's actual risk
profile**, not swept all at once and not copied from a generic tutorial default. For this
project, the biggest risks — in priority order — are:

1. Overfitting a tree to spurious in-sample ordering (capacity control)
2. Choosing a training/label scheme that doesn't match the trading objective (top-decile long-short)
3. Under- or over-weighting ranking-specific mechanics (pairwise gradient shape, gain function)
4. Fine-grained regularization (lowest priority — only tune after 1-3 are stable)

Every experiment should be evaluated on:
- **Walk-forward mean IC** (`summarize_ic`)
- **IC IR** (mean / std) — penalizes instability, more honest than mean IC alone
- **Train-vs-test IC gap** — currently not computed in `model.py`; add in-sample IC per
  fold (predict on `X_train`, compare IC to the OOS fold) so overfitting is visible
  directly, not inferred
- **Per-fold IC time series**, not just the aggregate — given the label dispersion shift
  noted above, a hyperparameter's effect on IC *stability* across regimes matters as much
  as its effect on the mean

---

## Prioritized Roadmap

### Tier 0 — Diagnostics before touching any hyperparameter
- [ ] Confirm training window sizes across all folds: `[len(w) for w, _ in walk_forward_splits(...)]` — verify the 58-month first fold converges to 60 and isn't a persistent embargo artifact.
- [ ] Add in-sample IC computation per fold (predict on `X_train`, compute IC against training labels) so every subsequent experiment already has a visible train/test IC gap.
- [ ] Build a lightweight experiment harness: given a hyperparameter override dict, run `run_walk_forward`, record mean IC / IC IR / train-test gap / per-fold IC series to a results table, so sweeps are comparable side by side rather than scattered across notebook cells.

### Tier 1 — Capacity control (tune first; highest-impact for this data)
Rationale: with ~476 names per month, the model has plenty of cross-sectional signal to
work with per fold — the risk isn't a thin query group, it's a tree memorizing noise
across a modest number of independent time periods (~35 OOS folds).

- [ ] `max_depth` — sweep {2, 3, 4, 5, 6}. Watch train/test IC gap, not just OOS IC.
- [ ] `min_child_weight` — sweep {5, 15, 30, 60}. Given the healthy, stable group sizes confirmed above, this does **not** need to be pushed unusually high by default — treat it as a normal regularization dial, contrary to the generic "small financial query groups" heuristic that turned out not to apply here.
- [ ] `subsample` / `colsample_bytree` — sweep {0.6, 0.7, 0.8, 1.0}. Column subsampling also decorrelates trees across the momentum/vol/size/liquidity factors, which are likely correlated with each other.

### Tier 2 — Learning dynamics (tune second)
- [ ] `learning_rate` × `n_estimators` jointly — compare e.g. (0.1, 50) vs (0.01, 500) at matched effective capacity, with early stopping against a walk-forward validation fold rather than a fixed round count.
- [ ] Confirm early stopping is wired to the *walk-forward* validation IC, not a random holdout — a random split would leak adjacent-time information.

### Tier 3 — Ranking-specific mechanics (tune third, once Tier 1-2 stable)
- [ ] `objective`: `"rank:ndcg"` vs `"rank:pairwise"` — direct ablation of the core ranking objective choice.
- [ ] Decile count in the `pd.qcut` label — try 5 and 20 in addition to the current 10. Tests whether label granularity changes how much signal LambdaMART's pairwise gradients can extract.
- [ ] `lambdarank_pair_method` (`mean` vs `topk`) — if the eventual trading strategy only acts on the top decile, `topk` focuses gradient signal on exactly the positions that drive P&L.
- [ ] `ndcg_exp_gain` (exponential vs linear gain) — with 10 relevance tiers, exponential gain (default) lets the top tier dominate the loss; linear gain may better match a "get the full ordering right" objective. Worth an explicit ablation rather than assuming the default is correct.

### Tier 4 — Fine-grained regularization (lowest priority)
- [ ] `reg_alpha`, `reg_lambda`, `gamma` — only tune after Tier 1 has already brought the train/test IC gap under control. Tuning these before capacity parameters is a common wasted-effort pattern.

### Cross-cutting: training window design
- [ ] `train_months` — sweep {36, 60, 120}. Tests the regime-relevance vs. sample-size tradeoff. Evaluate using the *per-fold* IC time series (not just the mean), since a shorter window may help in some regimes and hurt in others.

---

## Interview / Write-Up Framing

The strong answer to "how did you tune this" is not "I grid-searched N hyperparameters."
It's: *"I first measured my actual query-group size and panel length rather than assuming
generic financial-ranking heuristics applied. That told me overfitting risk was
concentrated in tree capacity given a modest number of independent walk-forward folds, not
in thin cross-sections — so I prioritized capacity-constraining parameters first, verified
the train/test IC gap closed before tuning ranking-specific mechanics, and evaluated every
change on the per-fold IC time series rather than a single aggregate number, since label
dispersion in this data visibly shifts across regimes."*

That reasoning chain — measure first, prioritize by verified risk, distrust an aggregate
metric — is what a hedge-fund-track interviewer is actually probing for in a "walk me
through your choices" conversation.
