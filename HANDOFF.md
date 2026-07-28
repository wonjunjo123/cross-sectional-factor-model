# Handoff: Point-in-Time Cross-Sectional Factor Model

## What this project is

A cross-sectional equity factor model that ranks S&P 500 stocks by
predicted next-month relative return. Compares a linear (Fama-MacBeth-
style) baseline against a gradient-boosted (LightGBM) model on the same
factor set, and backtests a long-short decile portfolio on the resulting
rankings. Built for quant research internship applications (hedge
fund / prop trading recruiting).

**Full title:** Point-in-Time Cross-Sectional Factor Model: Linear vs.
Gradient Boosting
**Short title (resume):** Point-in-Time Cross-Sectional Factor Model

## Current state — READ THIS FIRST

**The pipeline is written but has not been run end-to-end.** No real
results exist yet. Every file in `src/` compiles (verified with
`py_compile`) but has not been executed against real data. There is no
`output/model_comparison.csv` yet, no Sharpe ratio, no IC numbers.
**Do not fabricate or estimate placeholder results** — the user needs
real numbers from an actual run before finalizing any resume bullets or
write-up claims.

## Immediate next steps, in order

1. **Verify WRDS table/column names before running anything.** WRDS has
   restructured table naming before (Compustat's index-constituents table
   was pulled from WRDS in 2020; CRSP migrated some tables to a newer
   "CIZ" format around 2022). Connect interactively and run:
   ```python
   import wrds
   db = wrds.Connection(wrds_username="...")
   db.describe_table('crsp_a_indexes', 'dsp500list_v2')
   db.describe_table('crsp', 'dsf')
   ```
   Confirm these columns exist exactly as used in `data_prep.py`:
   `mbrstartdt`, `mbrenddt`, `permno` (membership table); `permno`,
   `date`, `prc`, `ret`, `vol`, `shrout`, `cfacpr`, `cfacshr` (price
   table). If `dsp500list_v2` doesn't exist, try `dsp500list` (older,
   same columns). If `crsp.dsf` doesn't exist, the newer CIZ-format
   equivalent is `crsp.dsf_v2` / `crsp.stksecurityinfohist` — that would
   require edits to the query in `data_prep.py`, not just a table name
   swap (different column names and a join against the security-history
   table are needed).

2. **Run the WRDS pull:**
   ```bash
   pip install pandas numpy scikit-learn scipy lightgbm wrds pyarrow
   python src/data_prep.py <wrds_username>
   ```
   This caches `data/sp500_membership.parquet` and `data/prices_wrds.parquet`.
   Expect this to take a while — batched in groups of 500 PERMNOs.

3. **Run the full pipeline:**
   ```bash
   python src/main.py
   ```
   Produces `output/feature_panel.parquet`, `output/predictions_linear.parquet`,
   `output/predictions_gbm.parquet`, and `output/model_comparison.csv`.

4. **Sanity-check the output before trusting it**, specifically:
   - Row counts per month in the feature panel should be roughly 400–505
     (S&P 500 size, with some attrition for stocks lacking full 12-month
     trailing history). If it's much lower, the membership filter or
     the dropna in `build_feature_panel` is probably too aggressive.
   - IC values (`model.summarize_ic`) should be small — realistic monthly
     cross-sectional IC for equity factors is typically in the 0.02–0.08
     range. An IC above ~0.15 is a red flag for a leak somewhere (most
     likely a look-ahead bug), not a good result — investigate before
     believing it.
   - Check `compute_turnover` output isn't empty/NaN — if the decile
     assignment or holdings tracking broke, this fails silently as an
     empty series rather than an error.

5. **Once real numbers exist**, update the resume bullets (template
   below, in "Resume bullets" section) with actual values — do not keep
   the bracketed placeholders.

## Key design decisions (and why — don't silently change these)

- **PERMNO, not ticker, is the primary key everywhere in the pipeline.**
  Tickers change (Facebook → Meta) and get reused after delisting.
  `data_prep.get_permno_ticker_map` exists only for labeling output for
  human readability — never use it as a join key.

- **Point-in-time membership filter is applied AFTER feature computation,
  BEFORE cross-sectional normalization.** This ordering is load-bearing,
  not stylistic — see the docstring on `features.filter_to_membership`
  and the module docstring in `features.py`. Filtering too early starves
  trailing-window features (e.g. 12-month momentum) of legitimate
  pre-membership lookback data. Filtering too late lets non-members leak
  into a date's cross-sectional z-scores. If you refactor this file,
  preserve the order: full-history feature computation → membership
  filter → cross-sectional normalization.

- **Momentum is 12-1, not 12-0** (`mom_12m_ex1` excludes the most recent
  month). This is deliberate: 1-month reversal is a distinct,
  opposite-signed effect from 12-month momentum, and mixing them muddies
  the signal. Don't "fix" this to be a clean 12-month window without
  understanding why it's currently excluded.

- **CRSP's `ret` field is used directly for returns**, not reconstructed
  from a price series. `ret` is CRSP's own total return (price change +
  dividends), delisting-adjusted by CRSP itself. Also: CRSP stores a
  **negative price** in `prc` when the value is a bid/ask midpoint
  estimate rather than an actual trade price — always `.abs()` it before
  using it for market cap or any price-level calculation (already done
  in `data_prep.py`; preserve this if you touch that code).

- **Market cap (`log_mkt_cap`) is a true size factor**, computed as
  `prc * shrout * 1000` (shrout is in thousands in CRSP). This replaced
  an earlier version of the project that used log(dollar volume) as a
  size proxy — that was flagged as not a real size factor. Don't
  reintroduce that substitution.

- **Walk-forward validation only, never a random train/test split.**
  See `model.walk_forward_splits` and its docstring for the reasoning on
  purge/embargo gaps — currently not implemented because the current
  feature set's lookback windows don't overlap the test boundary, but
  that assumption breaks if a shorter-window feature is added later
  (flagged explicitly in the docstring — check it before adding features).

- **IC (Spearman rank correlation), not R², is the primary evaluation
  metric.** This is what practitioners actually use for a ranking model,
  since it evaluates order, not magnitude — which is what matters for a
  long-short portfolio built on ranks.

## Known limitations (disclosed, not oversights — don't "fix" silently)

- **No fundamentals data.** Factors are price/volume-based only
  (momentum, volatility, size, liquidity). If fundamentals are added
  later (value, quality factors), they MUST be joined on as-reported /
  filing date, not fiscal period end date (`datadate` in Compustat) — a
  company's Q1 earnings aren't public knowledge until weeks after the
  fiscal period ends, and joining on period-end date is a serious
  look-ahead bug. This needs real design discussion before implementing,
  not a quick join.
- **No transaction cost model.** Turnover is reported explicitly instead,
  as the input needed to estimate cost impact, but costs aren't deducted
  from the reported Sharpe/returns.

## Repo structure

```
xsec_ml_project/
├── README.md          # detailed project description, run instructions
├── data/               # WRDS pull cache (parquet) — not in this handoff, generated on run
├── output/             # pipeline outputs — not yet generated
├── notebooks/          # empty, for exploratory analysis if useful
└── src/
    ├── data_prep.py    # WRDS pull: point-in-time membership + CRSP prices
    ├── features.py     # factor construction, membership filter, normalization
    ├── model.py         # walk-forward CV, linear + GBM models, IC scoring
    ├── backtest.py       # decile portfolio construction, performance metrics
    └── main.py           # orchestrates the full pipeline
```

## Resume bullets (template — fill in real numbers after running)

- Built point-in-time equity panel across [X] S&P 500 constituents
  (20XX–20XX) via WRDS/CRSP, eliminating survivorship bias from
  current-constituent backtests
- Engineered momentum, volatility, size, and liquidity factors; trained
  linear and gradient-boosted (LightGBM) ranking models via walk-forward
  cross-validation, evaluated on Information Coefficient
- Backtested long-short decile portfolio in Python, achieving [X] Sharpe
  and [X]% IC improvement over linear baseline, with turnover analysis
  for cost viability

## What NOT to do without discussing first

- Don't add fundamentals data with a naive `datadate` join (see look-ahead
  warning above).
- Don't switch back to a "current constituents" universe or yfinance data
  — the whole point of this version was fixing survivorship bias.
- Don't report Sharpe/IC numbers in the write-up or resume that weren't
  actually produced by a real run of `main.py`.
- Don't silently change the momentum exclusion window, the membership
  filter ordering, or the walk-forward (vs. random split) validation
  approach — each was a deliberate fix for a specific, named failure mode.
