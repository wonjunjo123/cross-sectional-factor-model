"""
data_prep.py

Point-in-time S&P 500 membership and CRSP daily price data via WRDS.

This replaces the earlier yfinance/current-constituents approach specifically
to fix survivorship bias: instead of using today's 500 tickers, this pulls
every stock that was EVER a member of the index during the sample window,
along with the exact date ranges each stock was a member, and uses CRSP's
PERMNO as the primary identifier. PERMNO, not ticker, is the right key here
-- tickers change (Facebook -> Meta), get reused after a company delists,
and are ambiguous across time. PERMNO is the one identifier CRSP guarantees
is stable for the life of a security.

Requires:
    pip install wrds
    A WRDS account with CRSP access (crsp_a_indexes, crsp libraries)

IMPORTANT -- verify before running: WRDS table names have shifted before
(Compustat's index-constituents table was pulled from WRDS in 2020; CRSP
migrated some tables to a newer "CIZ" format around 2022, e.g. crsp.dsf_v2
/ crsp.stksecurityinfohist alongside the legacy crsp.dsf). Run
`db.describe_table('crsp_a_indexes', 'dsp500list_v2')` and
`db.describe_table('crsp', 'dsf')` first to confirm these tables and
columns still exist in your WRDS instance exactly as used below. If
dsp500list_v2 isn't available, dsp500list (no _v2) has the same columns.
"""

import wrds
import pandas as pd
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def connect(wrds_username: str = None) -> wrds.Connection:
    """
    Opens a WRDS connection. Prompts for username/password on first call
    if not cached. WRDS supports a saved .pgpass file for passwordless
    connects after the first run -- see WRDS's Python setup docs if you're
    running this repeatedly.
    """
    return wrds.Connection(wrds_username=wrds_username)


def get_sp500_membership(db: wrds.Connection, start: str, end: str) -> pd.DataFrame:
    """
    Pulls point-in-time S&P 500 membership spells from CRSP's index list.

    Each row is a (permno, start, ending) membership SPELL -- a stock that
    left and later rejoined the index appears as two separate rows. This
    is the actual survivorship-bias fix: instead of "who is in the S&P 500
    today," this answers "who was in the S&P 500 on each historical date,"
    including names that no longer exist (acquisitions, delistings, index
    drops for falling market cap, etc).
    """
    query = f"""
        SELECT permno, mbrstartdt AS start, mbrenddt AS ending
        FROM crsp_a_indexes.dsp500list_v2
        WHERE mbrenddt >= '{start}' AND mbrstartdt <= '{end}'
    """
    membership = db.raw_sql(query, date_cols=["start", "ending"])
    membership["permno"] = membership["permno"].astype(int)
    return membership


def download_prices_wrds(
    db: wrds.Connection, permnos: list[int], start: str, end: str
) -> pd.DataFrame:
    """
    Pulls daily CRSP prices, returns, volume, and shares outstanding for
    the given PERMNOs. Downloaded in batches -- a single IN clause with
    the full multi-hundred-PERMNO list can be slow or time out; 500 at a
    time is a safe chunk for a universe this size.

    NOTE on `ret`: this is CRSP's own total return (price change plus
    dividends, delisting-adjusted by CRSP directly) -- more reliable than
    reconstructing returns from an adjusted-close price series, which is
    what the earlier yfinance version did. Use `ret` directly; don't
    recompute returns from `prc`.

    NOTE on `prc`: CRSP stores a NEGATIVE price when the closing trade
    price wasn't available and the value is a bid/ask midpoint estimate
    instead. Take the absolute value before using it for anything (market
    cap, price-level filters, etc) -- a well-known CRSP quirk that trips
    people up if they don't know to check for it.
    """
    frames = []
    batch_size = 500
    for i in range(0, len(permnos), batch_size):
        batch = permnos[i : i + batch_size]
        permno_list = ",".join(str(p) for p in batch)
        query = f"""
            SELECT permno, date, prc, ret, vol, shrout, cfacpr, cfacshr
            FROM crsp.dsf
            WHERE permno IN ({permno_list})
              AND date BETWEEN '{start}' AND '{end}'
        """
        df = db.raw_sql(query, date_cols=["date"])
        frames.append(df)
        print(f"  downloaded permno batch {i // batch_size + 1} ({len(df)} rows)")

    panel = pd.concat(frames, ignore_index=True)
    panel["prc"] = panel["prc"].abs()
    panel["mkt_cap"] = panel["prc"] * panel["shrout"] * 1000  # shrout is in thousands
    panel["dollar_vol"] = panel["prc"] * panel["vol"]
    panel = panel.sort_values(["permno", "date"]).reset_index(drop=True)

    DATA_DIR.mkdir(exist_ok=True)
    panel.to_parquet(DATA_DIR / "prices_wrds.parquet", index=False)
    return panel


def get_permno_ticker_map(db: wrds.Connection, permnos: list[int]) -> pd.DataFrame:
    """
    Optional readability layer: maps PERMNO to the ticker/company name
    that was current as of each namedt/nameenddt window, from
    crsp.stocknames. Useful for labeling output tables and plots so you're
    not reading raw PERMNOs in your write-up -- NOT used as a join key
    anywhere in the pipeline itself, since a ticker can map to different
    companies at different points in history.
    """
    permno_list = ",".join(str(p) for p in permnos)
    query = f"""
        SELECT permno, ticker, comnam, namedt, nameenddt
        FROM crsp.stocknames
        WHERE permno IN ({permno_list})
    """
    return db.raw_sql(query, date_cols=["namedt", "nameenddt"])


def build_universe_and_prices(
    wrds_username: str = None, start: str = "2012-01-01", end: str = "2026-01-01"
):
    """
    Orchestrates the full WRDS pull: membership spells first, then prices
    for every PERMNO that was ever a member during the window (not just
    today's constituents).
    """
    db = connect(wrds_username)

    print("Pulling point-in-time S&P 500 membership...")
    
    # pulls the list? df? of PERMNOs active during that time
    membership = get_sp500_membership(db, start, end)
    DATA_DIR.mkdir(exist_ok=True)
    membership.to_parquet(DATA_DIR / "sp500_membership.parquet", index=False)
    n_permnos = membership["permno"].nunique()
    print(
        f"  {n_permnos} unique PERMNOs were in the index at some point "
        f"during {start} to {end} ({len(membership)} membership spells -- "
        f"a stock that left and rejoined counts as 2+ spells)"
    )

    unique_permnos = membership["permno"].unique().tolist()

    print(f"Downloading CRSP daily prices for {len(unique_permnos)} PERMNOs...")
    prices = download_prices_wrds(db, unique_permnos, start, end)

    db.close()
    return membership, prices


if __name__ == "__main__":
    import sys

    start = '2024-01-01'
    end = '2026-01-01'

    username = sys.argv[1] if len(sys.argv) > 1 else None
    build_universe_and_prices(wrds_username=username, start=start, end=end)
