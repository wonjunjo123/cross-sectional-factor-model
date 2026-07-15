# Cross-Sectional ML Return Ranking

Predicts next-month relative equity returns across the S&P 500 using
cross-sectional factor features, compares a linear (Fama-MacBeth-style)
baseline against a gradient-boosted model, and backtests a long-short
decile portfolio built on the resulting rank predictions.

## Data source and survivorship bias

Universe and prices are sourced from **WRDS/CRSP**, not yfinance:

- **Point-in-time index membership**: `crsp_a_indexes.dsp500list_v2` gives
  exact membership spells (start/end dates) for every PERMNO that was
  *ever* in the S&P 500 during the sample window -- not just today's 500
  constituents. This is the actual survivorship-bias fix: the backtest
  universe on any given historical date includes stocks that have since
  been delisted, acquired, or dropped from the index, because they were
  genuinely investable on that date.
- **Identifier**: PERMNO, not ticker. Tickers change over time (Facebook
  -> Meta) and get reused after delisting; PERMNO is the identifier CRSP
  guarantees is stable for the life of a security. A ticker/company-name
  lookup (`get_permno_ticker_map`) is provided for labeling output only --
  it is never used as a join key in the pipeline itself.
- **Returns**: CRSP's own `ret` field (total return, dividend- and
  delisting-adjusted by CRSP directly), not a return reconstructed from
  an adjusted-close price series.
- **Market cap**: computed directly from CRSP `prc` x `shrout`, giving a
  true size factor rather than a liquidity-based stand-in.

### How the point-in-time filter is applied (read this before extending the pipeline)

Features are computed across each PERMNO's **full available price
history first** -- a stock needs trailing price data to compute momentum
correctly even in the months just before or after its actual index
membership window. The membership filter is applied **afterward**, right
before cross-sectional normalization. Get this order wrong in either
direction and you introduce a bug:

- Filter too early -> trailing-window features (12-month momentum, etc.)
  get starved of legitimate lookback data right at each stock's entry
  into the index.
- Filter too late (e.g. after normalization) -> stocks that weren't
  actually index members on a given date leak into that date's
  cross-sectional z-scores and the eventual portfolio.

`features.filter_to_membership` is the function that does this, and its
docstring explains the reasoning again in context.

## Remaining known limitations (deliberate, disclosed)

- **No fundamentals data** in this version -- factors are price/volume-
  based only (momentum, realized volatility, size, liquidity). A
  fundamentals-based value or quality factor is a natural extension, but
  needs as-reported (not fiscal-period-end) timing to avoid look-ahead
  leakage -- don't add this casually.
- **Transaction costs are not modeled directly.** Turnover is reported
  explicitly instead, as the input needed to estimate cost impact.

## Pipeline

1. `data_prep.py` -- pulls point-in-time S&P 500 membership and CRSP
   daily prices from WRDS, caches both to parquet
2. `features.py` -- resamples to monthly, builds momentum/vol/size/
   liquidity factors from full price history, applies the point-in-time
   membership filter, cross-sectionally z-scores, builds the forward-
   return target
3. `model.py` -- walk-forward (rolling-window) cross-validation; trains a
   linear baseline and a LightGBM model on each window; scores
   predictions using the Information Coefficient (Spearman rank
   correlation)
4. `backtest.py` -- builds a long top-decile / short bottom-decile
   portfolio from each model's predictions; computes annualized return,
   volatility, Sharpe, max drawdown, and turnover; compares models
   side-by-side
5. `main.py` -- runs the full pipeline end to end

## Run it

```bash
pip install pandas numpy scikit-learn scipy lightgbm wrds pyarrow

python src/data_prep.py <your_wrds_username>   # one-time WRDS pull
python src/main.py                             # features -> models -> backtest -> comparison
```

`data_prep.py` will prompt for your WRDS password on first run if it
isn't cached. **Before running for the first time**, connect to WRDS
interactively and run:

```python
db.describe_table('crsp_a_indexes', 'dsp500list_v2')
db.describe_table('crsp', 'dsf')
```

to confirm these tables and the columns used in `data_prep.py`
(`mbrstartdt`, `mbrenddt`, `permno`, `prc`, `ret`, `vol`, `shrout`,
`cfacpr`, `cfacshr`) still match your WRDS instance. CRSP has
restructured table naming before (the newer CIZ-format tables are
`crsp.dsf_v2` / `crsp.stksecurityinfohist`) -- don't assume the legacy
names used here are still current without checking.

## What the write-up should center on

Not "the model made money" -- the comparison table in
`output/model_comparison.csv`. The actual finding is a calibrated claim:
does the nonlinear model improve IC/Sharpe over the linear baseline, and
at what turnover cost? That trade-off, stated plainly, is the
three-minute interview pitch -- and now you can add a second sentence
about how the point-in-time universe construction avoids overstating
the result, which is exactly the kind of detail that separates a
defensible backtest from a leaked one in an interview.

## Natural extensions (don't build these unless you have spare time)

- Fundamentals-based factors (value, quality) with as-reported timing
  to avoid look-ahead leakage
- Explicit transaction cost model applied to the turnover series
- Purge/embargo gap in the walk-forward split, if a shorter-window
  feature is added that could overlap the test period's information set
