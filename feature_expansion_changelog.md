# Feature Expansion Changelog

What was built against `feature_expansion_instructions.md`, and why. This covers
**Phase 0 only** (everything buildable without real WRDS fundamentals data) — see
"What's blocked" at the bottom for what's left.

Full design rationale lives in the plan this was built from:
`~/.claude/plans/immutable-floating-emerson.md`. This document is the shorter,
after-the-fact "what actually happened" version, including bugs caught during
implementation that the plan didn't anticipate.

---

## Two things that changed the instructions doc's assumptions

1. **`data_prep.py` doesn't exist.** It was archived to `api_data_prep.py` (commit
   `5f05a89`, "archiving data_prep as we don't use API anymore"). The actual working
   convention for pulling WRDS data is `load_wrds_web_export.py` — manual CSV export
   from the WRDS website, not the live `wrds` Python API.
2. **Live WRDS API auth doesn't work in this session either.** Tested directly:
   `wrds.Connection()` hangs on an interactive username prompt and fails with `EOFError`
   non-interactively. Confirmed with you that new pulls should follow the
   `load_wrds_web_export.py` convention (manual web-query CSV exports you run yourself)
   rather than resurrecting the API path.

## New files

### `src/pit_merge.py`
One reusable point-in-time asof-merge function, `merge_asof_pit`, used by every
fundamental feature. Centralizes the "was this public by the feature date?" check in
one place instead of three separate hand-rolled merges.

**Two pandas behaviors I verified empirically rather than assumed** (both would have
caused either a crash or a silent data-corruption bug):
- `pd.merge_asof(..., by=...)` requires the `on` column sorted **globally**, not
  `[by, on]` composite — the composite sort raises `ValueError: left keys must be
  sorted` the moment permnos interleave in date order, which they always do in this
  panel. Confirmed by reproducing the error, then confirming the global-sort fix works.
- `pd.merge_asof` always returns a **fresh RangeIndex**, discarding the input's index,
  even though it preserves row order exactly. `merge_asof_pit` restores the original
  index (`merged.index = left_sorted.index`, then `.sort_index()`) before returning —
  without this, `add_analyst_revision_features` (which combines two separate
  `merge_asof_pit` calls via index-aligned Series subtraction) would silently misalign
  rows with no error, since two nearby permno-date rows could easily get swapped without
  Python ever noticing.

Verified against a hand-built synthetic fixture (multi-permno, interleaved dates, two
independent asof references 3 months apart) — every value matched manual calculation.

### `src/baseline.py`
Fama-MacBeth-style linear baseline (validation checklist item 5). No `statsmodels`
dependency (not installed; not worth adding for one OLS) — coefficients/t-stats via
`numpy.linalg.lstsq` and the normal equations directly.

- `fama_macbeth_coefs` / `summarize_fama_macbeth`: classic per-date cross-sectional OLS,
  averaged across time. Reports **two** t-stats side by side (avg of each month's own
  t-stat, and the textbook Fama-MacBeth `mean(coef)/(std(coef)/sqrt(T))`) since they
  answer different questions and neither is obviously "the" default.
- `run_linear_walk_forward`: out-of-sample counterpart, mirrors `model.run_walk_forward`'s
  exact fold structure (same `walk_forward_splits` call, same embargo, same train/test
  extraction, same per-month target winsorization) so folds line up 1:1 with the GBM's.
  Returns the identical `[date, permno, fwd_ret, pred]` shape, so `model.summarize_ic()`
  runs on it unmodified — no baseline-specific IC code needed.

**Bug found and fixed during testing:** running this against the real
`output/feature_panel.parquet` (with the old 6-feature set, since the new fundamentals
aren't in that cached file yet) initially threw a wall of `RuntimeWarning: divide by
zero / overflow / underflow encountered in matmul`. Traced it directly: manually
inspected `resid`, `sigma2`, and `pinv` output at the exact warning site and found every
value was finite and correct — this is a known false-positive quirk of macOS's
Accelerate BLAS backend on certain matmul calls, not an actual numerical problem.
Fixed by scoping `np.errstate(...)` suppression tightly around just the matmul calls in
`_ols_with_tstats` and `run_linear_walk_forward` (not a global suppression, so a
genuinely bad value elsewhere would still warn normally). Confirmed clean with
`warnings.filterwarnings("error")` — zero warnings, identical output values before and
after the fix.

### `src/load_wrds_fundamentals.py`
Mirrors `load_wrds_web_export.py`'s shape: docstring specifying exact WRDS web-query
exports needed → column-mapping dicts → `build_*` loaders → `main()`.

**7 manual CSV exports needed from you** (vs. the 1 you've done before for CRSP) — table
names are best-effort, not verified, same caveat `api_data_prep.py` already carries
about WRDS table drift:

| # | Purpose | Table (verify at pull time) |
|---|---|---|
| 1 | FY1 + next-qtr consensus | `ibes.statsumu_epsus` |
| 2 | Earnings actuals + announce date | `ibes.surpsumu_epsus` |
| 3 | IBES ticker→PERMNO link | `wrdsapps_ibcrsphist.ibcrsphist` |
| 4 | Short interest | `comp.sec_shortint` |
| 5 | GVKEY↔PERMNO link | `crsp.ccmxpf_lnkhist` |
| 6 | GICS sector | `comp.company` |
| 7 | SIC fallback | `crsp.stksecurityinfohist` |

Full field lists and CSV filename patterns are in the module's docstring. **#6/#7
(sector) are the most skippable if this is a lot at once** — Features 1–3 don't depend
on them, so sector neutralization could be deferred without blocking anything else.

Every `build_*` function was tested against small hand-built fixtures matching the
documented raw-export schema (ticker/GVKEY→PERMNO resolution via date-validity windows,
not flat identifier maps) — all passed.

### `src/validate_features.py`
Implements the 5-item validation checklist as independently re-runnable functions:
`coverage_report`, `leakage_spot_check`, `timing_regression_test`,
`rerun_pipeline_smoke_test`, `baseline_comparison`. Structurally complete; `coverage_report`
and `leakage_spot_check` were tested against a synthetic fixture (see below) — the other
three reuse already-verified pieces (`build_feature_panel`, `run_walk_forward`,
`baseline.run_linear_walk_forward`) and are ready to run once real data lands.

**Bug found and fixed during testing:** `leakage_spot_check`'s SUE trace initially
crashed with `KeyError: 'sue'` — I'd assumed `features._build_quarterly_surprises`
produces a `sue` column, but it only computes the raw `surprise` (actual − consensus);
the 3-tier std-normalization into `sue` happens in `add_sue_feature` itself. Rather than
re-implementing that tiering logic a second time in the validation script (which would
let the two copies silently drift apart), I split `add_sue_feature` in `features.py`
into a reusable `_add_sue_column` helper that both `add_sue_feature` and
`leakage_spot_check` now call.

## Modified files

### `src/features.py`
- Added `from __future__ import annotations` — needed because this file didn't have it
  (unlike `model.py`/`tune.py`), and the new `X | None` type hints don't parse on
  Python 3.9 without it. Caught immediately by the first test run.
- `resample_to_monthly`: added `month_end_prc` and `shrout` to the aggregation — these
  weren't being carried into the monthly panel at all before, but `est_rev_3m` needs the
  price denominator and `short_ratio` needs the share count. Found this reading the
  function closely during planning, not called out in the instructions doc itself.
- New functions: `add_analyst_revision_features`, `_build_quarterly_surprises` +
  `_add_sue_column` + `add_sue_feature`, `add_short_interest_feature`, `attach_sector`
  (+ a static `_ff12_from_siccd` SIC-range lookup — see caveat below).
- `cross_sectional_normalize`: new `sector_col`/`min_group_size` params, `None` by
  default (reproduces prior behavior exactly). When given, blends per-`(date, sector)`
  and full-cross-section z-scores via `np.where(count >= min_group_size, sector_z,
  full_z)`, where `count` is the **non-null count of that specific feature**, not group
  row count — verified with a dedicated test (12-name sector gets within-sector
  z-scores, a 4-name sector correctly falls back to the full cross-section, confirmed
  against a manual winsorize+z-score recomputation).
- `build_feature_panel`: gained 5 new required params (the raw fundamental dataframes).
  New pipeline order: momentum/vol/size → revision/SUE/short-interest (before
  `filter_to_membership`, so trailing windows like SUE's 8-quarter history aren't
  starved) → forward return → `filter_to_membership` → `attach_sector` (after the
  membership filter — sector has no lookback window) → sector-neutral normalize →
  coverage-gated missingness indicators → fundamentals-only NaN-fill-to-0 → the
  narrowed `dropna` (only `fwd_ret` + the 4 price-derived z-features).

**Bug found and fixed during testing (the most important one):** my first pass filled
**all** `z_cols` to 0 before the `dropna` step — including the 4 required price
features. That made the subsequent `dropna(subset=required)` completely vacuous: a row
with genuinely missing price data (e.g. insufficient trailing history) would get
silently kept as a zero instead of dropped, contradicting the instructions' explicit
"require non-null for fwd_ret and the four price-derived z-features only." Caught this
by running the synthetic end-to-end test and noticing the row count didn't drop as
expected. Fixed by splitting `feature_cols` into `price_cols` (never filled, still
gated by the `required` dropna) and `fundamental_cols` (the only ones that get the
fill-to-0 treatment).

### `src/model.py`
`FEATURE_COLS` updated to the instructions' exact closed list:
`["mom_3m_z", "mom_12m_ex1_z", "realized_vol_z", "log_mkt_cap_z", "est_rev_3m_z", "rev_breadth_z", "sue_z", "short_ratio_z"]`.
`mom_1m_z` and `log_dollar_vol_z` are dropped from the model's inputs but still computed
in `features.py` (per instructions Section 3). Missingness `_missing` indicator columns
are computed and stored but **not** added here — the instructions give an exact 8-column
list with no room for them, so wiring them in wasn't authorized by what was asked.

### `src/visualize.py`
Added a `"baseline"` key to `COLORS`/`MODEL_LABELS`. `plot_ic_timeseries` already
supported multi-model comparison (`dict of {model_name: DataFrame[date, ic]}`,
confirmed by reading it directly) — zero changes needed to the plotting function itself.
Verified end-to-end: generated a real GBM-vs-linear-baseline IC comparison chart from
the actual cached feature panel.

## Interpretive decisions (the instructions doc was ambiguous here — flagging, not hiding)

- **SUE denominator fallback**: the doc says both "fall back to consensus stdev if
  surprise history is short" and "require ≥4 prior quarters" — only consistent as a
  3-tier rule: ≥8 prior quarters → own rolling 8Q surprise std; 4–7 → cross-analyst
  consensus STDEV; <4 → NaN (hard gate, per the literal "require ≥4").
- **Sector fallback**: CCM+GICS `gsector` is primary; CRSP `siccd` → Fama-French 12
  industry is the fallback for PERMNOs with no valid CCM link; anything FF12 itself
  can't classify maps to "other."

## Known caveats worth your attention

- **FF12 SIC ranges** (`features.py`'s `_FF12_RANGES`): I don't trust my own memory of
  these exact numeric boundaries (a wrong one silently misclassifies names), so I fetched
  them from a maintained open-source replication
  (`github.com/ed-dehaan/FamaFrenchIndustries`) rather than transcribing from memory —
  and this actually caught a real error in my initial recollection (I'd misremembered
  `2800-2829` as belonging to "Manuf" when it's actually "Chems"). Still worth a final
  cross-check against Ken French's own `Siccodes12.zip` before trusting this in a
  published result, since I'm relying on a third-party mirror, not the primary source
  directly (the primary source is only distributed as a zip, which I can't fetch/parse
  as HTML).
- **WRDS table/field names in `load_wrds_fundamentals.py`** are best-effort, not
  verified against a live WRDS session — expect to need to adjust a table or column name
  per the module's own docstring guidance (search the WRDS web query browser for the
  current name, don't change the timing logic).
- `src/tune.py`'s `load_feature_panel()` still calls the **old** 2-arg
  `build_feature_panel(daily_panel, membership, horizon=HORIZON)` signature — this will
  break with a `TypeError` as soon as anyone runs it, since `build_feature_panel` now
  requires 5 more params. Deliberately left unfixed for now (it's Phase 2 work per the
  plan — mechanical, needs the same new `pd.read_parquet` loads as `main.py`). Likewise
  `main.py` itself hasn't been wired to the new `build_feature_panel` signature yet, so
  `python src/main.py` will also fail until Phase 2. This is expected, not an oversight —
  wiring both in now would just mean they immediately break again on missing fundamentals
  parquet files that don't exist yet.

## What's blocked (Phase 1, needs you)

Run the 7 WRDS web-query exports into `data/raw/` per `load_wrds_fundamentals.py`'s
docstring (or a reduced set, deferring sector — see the scope note above).

## What's next (Phase 2, after the CSVs land)

1. `python src/load_wrds_fundamentals.py` — inspect real row counts / PERMNO match
   rates, iterate on schema drift (expected).
2. Wire `main.py` and `tune.py` to the new `build_feature_panel` signature (mechanical).
3. `python src/main.py` end-to-end, then `python src/validate_features.py`, and iterate
   on whatever the checklist surfaces — including reporting a null result honestly if the
   GBM doesn't beat the linear baseline, per the instructions' explicit framing.
