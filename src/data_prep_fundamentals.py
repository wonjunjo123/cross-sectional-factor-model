"""
data_prep_fundamentals.py

Live wrds.Connection() pull for the 5 fundamental sources added by
feature_expansion_instructions.md. Originally planned as a manual
web-export (see load_wrds_fundamentals.py), because a first connection
attempt in this environment hung on an interactive username prompt and
was assumed broken -- turned out to just need `wrds_username` passed
explicitly to skip the prompt and use the cached .pgpass credentials.
Confirmed live and used to verify the exact table names below via
db.describe_table() before writing any of the queries here.

This script owns the SQL/identifier-resolution layer only. It reuses
load_wrds_fundamentals.py's build_*() functions unchanged for the actual
point-in-time-link resolution (ticker/GVKEY -> PERMNO via date-validity
windows) -- those were already written and tested against synthetic
fixtures; duplicating that logic here instead of importing it would let
the two copies silently drift apart.

Table names (verified live via db.describe_table() on 2026-08-16):
    ibes.statsumu_epsus            -- IBES unadjusted summary (FY1 + QTR consensus)
    ibes.actu_epsus                -- IBES unadjusted actuals
    wrdsapps_link_crsp_ibes.ibcrsphist -- IBES ticker <-> PERMNO link
    comp.sec_shortint               -- Compustat short interest
    crsp.ccmxpf_lnkhist              -- CRSP/Compustat Merged link
    comp.company                     -- GVKEY -> GSECTOR
    crsp.stksecurityinfohist          -- PERMNO -> SICCD history

Usage:
    python src/data_prep_fundamentals.py
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import wrds

from load_wrds_fundamentals import (
    build_ibes_actuals,
    build_ibes_estimates,
    build_sector_crosswalk,
    build_short_interest,
    build_siccd_history,
)

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
START = "2009-01-01"  # 2yr buffer before price data starts (2011-01-03), for
END = "2026-04-01"    # SUE's 8-quarter trailing history / 3mo-ago revisions
                       # to have something to match even in early test months.


def connect(wrds_username: str = "wonjunjo") -> wrds.Connection:
    return wrds.Connection(wrds_username=wrds_username)


def _sql_in(values: list) -> str:
    """Renders a Python list as a SQL IN-clause literal list, quoting
    strings. Small helper, not a general SQL-safety layer -- values here
    always come from WRDS's own PERMNO/TICKER/GVKEY columns (already
    validated identifiers pulled from a prior query), never from
    unvalidated external/user input."""
    if not values:
        return "(NULL)"
    if isinstance(values[0], str):
        return "(" + ",".join(f"'{v}'" for v in values) + ")"
    return "(" + ",".join(str(v) for v in values) + ")"


def pull_ibes_crsp_link(db: wrds.Connection, permnos: list[int]) -> pd.DataFrame:
    query = f"""
        SELECT ticker, permno, sdate, edate, score
        FROM wrdsapps_link_crsp_ibes.ibcrsphist
        WHERE permno IN {_sql_in(permnos)} AND score <= 2
    """
    df = db.raw_sql(query, date_cols=["sdate", "edate"])
    df["permno"] = df["permno"].astype(int)
    return df


def pull_ccm_link(db: wrds.Connection, permnos: list[int]) -> pd.DataFrame:
    query = f"""
        SELECT gvkey, lpermno AS permno, linkdt, linkenddt, linktype, linkprim
        FROM crsp.ccmxpf_lnkhist
        WHERE lpermno IN {_sql_in(permnos)}
          AND linktype IN ('LU','LC') AND linkprim IN ('P','C')
    """
    df = db.raw_sql(query, date_cols=["linkdt", "linkenddt"])
    df["permno"] = df["permno"].astype(int)
    return df


def pull_ibes_summary(db: wrds.Connection, tickers: list[str]) -> pd.DataFrame:
    query = f"""
        SELECT ticker, statpers, fpi, fpedats, measure, numest, numup, numdown, meanest, stdev
        FROM ibes.statsumu_epsus
        WHERE ticker IN {_sql_in(tickers)} AND measure = 'EPS' AND fpi IN ('1','6')
          AND statpers BETWEEN '{START}' AND '{END}'
    """
    return db.raw_sql(query, date_cols=["statpers", "fpedats"])


def pull_ibes_actuals(db: wrds.Connection, tickers: list[str]) -> pd.DataFrame:
    query = f"""
        SELECT ticker, anndats, pends AS fpedats, value AS actual, pdicity, measure
        FROM ibes.actu_epsus
        WHERE ticker IN {_sql_in(tickers)} AND measure = 'EPS' AND pdicity = 'QTR'
          AND anndats BETWEEN '{START}' AND '{END}'
    """
    return db.raw_sql(query, date_cols=["anndats", "fpedats"])


def pull_short_interest(db: wrds.Connection, gvkeys: list[str]) -> pd.DataFrame:
    query = f"""
        SELECT gvkey, datadate AS settlement_date, shortintadj AS shortint
        FROM comp.sec_shortint
        WHERE gvkey IN {_sql_in(gvkeys)}
          AND datadate BETWEEN '{START}' AND '{END}'
    """
    return db.raw_sql(query, date_cols=["settlement_date"])


def pull_company(db: wrds.Connection, gvkeys: list[str]) -> pd.DataFrame:
    query = f"SELECT gvkey, gsector FROM comp.company WHERE gvkey IN {_sql_in(gvkeys)}"
    return db.raw_sql(query)


def pull_siccd_history(db: wrds.Connection, permnos: list[int]) -> pd.DataFrame:
    query = f"""
        SELECT permno, siccd, secinfostartdt AS sic_start, secinfoenddt AS sic_end
        FROM crsp.stksecurityinfohist
        WHERE permno IN {_sql_in(permnos)}
    """
    df = db.raw_sql(query, date_cols=["sic_start", "sic_end"])
    df["permno"] = df["permno"].astype(int)
    return df


def main():
    membership = pd.read_parquet(DATA_DIR / "sp500_membership.parquet")
    permnos = membership["permno"].unique().tolist()
    print(f"Pulling fundamentals for {len(permnos)} PERMNOs, {START} to {END}...")

    db = connect()

    print("Resolving IBES ticker link...")
    ibes_link = pull_ibes_crsp_link(db, permnos)
    tickers = ibes_link["ticker"].dropna().unique().tolist()
    print(f"  {len(tickers)} distinct tickers across {ibes_link['permno'].nunique()} PERMNOs")

    print("Resolving CCM (GVKEY) link...")
    ccm_link = pull_ccm_link(db, permnos)
    gvkeys = ccm_link["gvkey"].dropna().unique().tolist()
    print(f"  {len(gvkeys)} distinct GVKEYs across {ccm_link['permno'].nunique()} PERMNOs")

    print("Pulling IBES summary estimates...")
    raw_summary = pull_ibes_summary(db, tickers)
    print(f"  {len(raw_summary)} raw rows")
    ibes_estimates = build_ibes_estimates(raw_summary, ibes_link)
    ibes_estimates.to_parquet(DATA_DIR / "ibes_estimates.parquet", index=False)
    print(f"  -> {len(ibes_estimates)} rows, {ibes_estimates['permno'].nunique()} PERMNOs matched"
          f" -> data/ibes_estimates.parquet")

    print("Pulling IBES actuals...")
    raw_actuals = pull_ibes_actuals(db, tickers)
    print(f"  {len(raw_actuals)} raw rows")
    ibes_actuals = build_ibes_actuals(raw_actuals, ibes_link)
    ibes_actuals.to_parquet(DATA_DIR / "ibes_actuals.parquet", index=False)
    print(f"  -> {len(ibes_actuals)} rows, {ibes_actuals['permno'].nunique()} PERMNOs matched"
          f" -> data/ibes_actuals.parquet")

    print("Pulling short interest...")
    raw_shortint = pull_short_interest(db, gvkeys)
    print(f"  {len(raw_shortint)} raw rows")
    short_interest = build_short_interest(raw_shortint, ccm_link)
    short_interest.to_parquet(DATA_DIR / "short_interest.parquet", index=False)
    print(f"  -> {len(short_interest)} rows, {short_interest['permno'].nunique()} PERMNOs matched"
          f" -> data/short_interest.parquet")

    print("Pulling sector (GICS)...")
    company = pull_company(db, gvkeys)
    sector_crosswalk = build_sector_crosswalk(company, ccm_link)
    sector_crosswalk.to_parquet(DATA_DIR / "sector_crosswalk.parquet", index=False)
    print(f"  -> {len(sector_crosswalk)} rows, {sector_crosswalk['permno'].nunique()} PERMNOs"
          f" -> data/sector_crosswalk.parquet")

    print("Pulling SIC history (sector fallback)...")
    raw_siccd = pull_siccd_history(db, permnos)
    siccd_history = build_siccd_history(raw_siccd)
    siccd_history.to_parquet(DATA_DIR / "siccd_history.parquet", index=False)
    print(f"  -> {len(siccd_history)} rows, {siccd_history['permno'].nunique()} PERMNOs"
          f" -> data/siccd_history.parquet")

    db.close()
    print("\nAll 5 fundamental sources pulled and cached.")


if __name__ == "__main__":
    main()
