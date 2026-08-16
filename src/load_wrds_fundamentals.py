"""
load_wrds_fundamentals.py

Alternative to a live wrds.Connection() pull for the 4 new fundamental
sources (IBES analyst estimates/actuals, Compustat short interest, GICS
sector) added by feature_expansion_instructions.md -- same reasoning as
load_wrds_web_export.py already established for CRSP prices/membership:
live `wrds` API auth doesn't work non-interactively in this environment
(confirmed: wrds.Connection() hangs on an interactive username prompt).
Builds cached, PERMNO-keyed parquet files in data/ from CSVs exported
manually via the WRDS website's web-based query tool, so features.py
needs no WRDS-connectivity code at all -- it only ever reads parquet.

WRDS TABLE NAMES: 5 of 7 verified directly against a live WRDS session
(see data_prep_fundamentals.py, which ended up being the path actually
used once the live API turned out to work after all -- confirmed via
db.describe_table() on 2026-08-16). The 2 below that were wrong guesses
are now corrected to their verified real names; kept this script and its
manual-export path around as the documented fallback for a future
session where the live API genuinely doesn't work.

Expected raw exports in data/raw/ (7 total -- more than the 1 you've done
before for CRSP; #6/#7 are the most skippable if this is a lot at once,
since sector neutralization is the lowest-priority feature and nothing
else in the pipeline depends on it):

  1. IBES Estimates -> Summary Statistics -> Unadjusted Summary (US)
     table (verified): ibes.statsumu_epsus
     fields: TICKER, STATPERS, FPI, FPEDATS, MEASURE, NUMEST, NUMUP,
             NUMDOWN, MEANEST, STDEV
     filter: MEASURE='EPS', FPI in ('1','6')  (1=FY1, 6=next fiscal qtr)
     -> data/raw/*ibes_summary*.csv

  2. IBES Estimates -> Actuals (unadjusted)
     table (verified): ibes.actu_epsus (NOT ibes.surpsumu_epsus -- that
     table doesn't exist; surpsumu is a different derived-surprise
     product, not the raw actuals+anndats this pipeline needs)
     fields: TICKER, ANNDATS, PENDS (-> fpedats), VALUE (-> actual),
             PDICITY, MEASURE
     filter: MEASURE='EPS', PDICITY='QTR'
     -> data/raw/*ibes_actuals*.csv

  3. IBES-CRSP link (ICLINK)
     table (verified): wrdsapps_link_crsp_ibes.ibcrsphist (NOT
     wrdsapps_ibcrsphist.ibcrsphist -- wrong library name)
     fields: TICKER, PERMNO, SDATE, EDATE, SCORE
     filter: SCORE <= 2  (best-quality links)
     -> data/raw/*ibes_crsp_link*.csv

  4. Compustat North America -> Supplemental -> Short Interest
     table (verified): comp.sec_shortint
     fields: GVKEY, IID, DATADATE, SHORTINTADJ
     -> data/raw/*short_interest*.csv

  5. CRSP/Compustat Merged -> Linking Table
     table (verified): crsp.ccmxpf_lnkhist
     fields: GVKEY, LPERMNO, LINKDT, LINKENDDT, LINKTYPE, LINKPRIM
     filter: LINKTYPE in ('LU','LC'), LINKPRIM in ('P','C')  (standard
             CCM merge convention -- primary, research-quality links)
     -> data/raw/*ccm_link*.csv

  6. Compustat North America -> Company file
     table (verified): comp.company
     fields: GVKEY, GSECTOR
     -> data/raw/*compustat_company*.csv

  7. CRSP -> Stock/Security Files -> Stock Header / Names History
     table (verified): crsp.stksecurityinfohist
     fields: PERMNO, SICCD, SECINFOSTARTDT (-> sic_start),
             SECINFOENDDT (-> sic_end)
     -> data/raw/*siccd_history*.csv

Cached outputs (all PERMNO-keyed -- ticker/GVKEY crosswalk resolution
happens entirely in this script; features.py never touches a ticker or
GVKEY):
    data/ibes_estimates.parquet   [permno, statpers, fpedats, period_type, meanest, stdev, numest, numup, numdown]
    data/ibes_actuals.parquet     [permno, anndats, fpedats, actual]
    data/short_interest.parquet   [permno, settlement_date, shortint]
    data/sector_crosswalk.parquet [permno, linkdt, linkenddt, gsector]
    data/siccd_history.parquet    [permno, sic_start, sic_end, siccd]

Usage:
    python src/load_wrds_fundamentals.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
RAW_DIR = DATA_DIR / "raw"

_IBES_SUMMARY_COLS = {
    "TICKER": "ticker", "STATPERS": "statpers", "FPI": "fpi",
    "FPEDATS": "fpedats", "MEASURE": "measure", "NUMEST": "numest",
    "NUMUP": "numup", "NUMDOWN": "numdown", "MEANEST": "meanest", "STDEV": "stdev",
}
_IBES_ACTUALS_COLS = {
    # ibes.actu_epsus's own field names are PENDS/VALUE, not FPEDATS/
    # ACTUAL -- renamed here so downstream code sees the same pipeline
    # names regardless of which raw table version supplied them.
    "TICKER": "ticker", "ANNDATS": "anndats", "PENDS": "fpedats",
    "VALUE": "actual", "PDICITY": "pdicity", "MEASURE": "measure",
}
_IBES_LINK_COLS = {
    "TICKER": "ticker", "PERMNO": "permno", "SDATE": "sdate",
    "EDATE": "edate", "SCORE": "score",
}
_SHORTINT_COLS = {
    "GVKEY": "gvkey", "DATADATE": "settlement_date", "SHORTINTADJ": "shortint",
}
_CCM_LINK_COLS = {
    "GVKEY": "gvkey", "LPERMNO": "permno", "LINKDT": "linkdt",
    "LINKENDDT": "linkenddt", "LINKTYPE": "linktype", "LINKPRIM": "linkprim",
}
_COMPANY_COLS = {"GVKEY": "gvkey", "GSECTOR": "gsector"}
_SICCD_COLS = {
    "PERMNO": "permno", "SICCD": "siccd",
    "SECINFOSTARTDT": "sic_start", "SECINFOENDDT": "sic_end",
}


def _find_one(pattern: str) -> Path:
    files = sorted(RAW_DIR.glob(pattern))
    if not files:
        raise FileNotFoundError(f"no {pattern} found in {RAW_DIR}")
    if len(files) > 1:
        print(f"  note: multiple files match {pattern}, concatenating: {[f.name for f in files]}")
    return files


def _load(pattern: str, col_map: dict) -> pd.DataFrame:
    frames = []
    for f in _find_one(pattern):
        df = pd.read_csv(f, usecols=list(col_map.keys()), low_memory=False)
        df = df.rename(columns=col_map)
        frames.append(df)
        print(f"  loaded {f.name} ({len(df)} rows)")
    return pd.concat(frames, ignore_index=True)


def _resolve_pit_link(
    raw: pd.DataFrame, link: pd.DataFrame, date_col: str, key_col: str,
    start_col: str, end_col: str, target_col: str,
) -> pd.DataFrame:
    """
    Resolves `raw`'s `key_col` identifier (a TICKER or GVKEY) to
    `target_col` (PERMNO) via `link`'s date-VALIDITY window
    [start_col, end_col] as of `raw`'s own `date_col` -- NOT a flat
    identifier map, since tickers and GVKEYs get reassigned across
    companies over time (see load_wrds_web_export.py's PERMNO-vs-ticker
    docstring note; the same reasoning applies to GVKEY). Rows with no
    valid link on that date are dropped -- a crosswalk failure, not a
    timing question, and features.py's job starts only after every
    source is already PERMNO-keyed.
    """
    r = raw.sort_values(date_col)
    lk = link.sort_values(start_col)
    merged = pd.merge_asof(r, lk, left_on=date_col, right_on=start_col,
                            by=key_col, direction="backward")
    valid = merged[target_col].notna() & (
        merged[end_col].isna() | (merged[date_col] <= merged[end_col])
    )
    return merged.loc[valid].reset_index(drop=True)


def build_ibes_estimates(raw_summary: pd.DataFrame, iclink: pd.DataFrame) -> pd.DataFrame:
    est = raw_summary[raw_summary["measure"] == "EPS"].copy()
    est["period_type"] = est["fpi"].map({"1": "FY1", "6": "QTR"})
    est = est.dropna(subset=["period_type"])
    est["statpers"] = pd.to_datetime(est["statpers"])
    iclink = iclink.copy()
    iclink["sdate"] = pd.to_datetime(iclink["sdate"])
    iclink["edate"] = pd.to_datetime(iclink["edate"])

    resolved = _resolve_pit_link(est, iclink, "statpers", "ticker", "sdate", "edate", "permno")
    resolved["permno"] = resolved["permno"].astype(int)
    resolved["fpedats"] = pd.to_datetime(resolved["fpedats"])
    return resolved[["permno", "statpers", "fpedats", "period_type",
                      "meanest", "stdev", "numest", "numup", "numdown"]]


def build_ibes_actuals(raw_actuals: pd.DataFrame, iclink: pd.DataFrame) -> pd.DataFrame:
    act = raw_actuals[(raw_actuals["measure"] == "EPS") & (raw_actuals["pdicity"] == "QTR")].copy()
    act["anndats"] = pd.to_datetime(act["anndats"])
    iclink = iclink.copy()
    iclink["sdate"] = pd.to_datetime(iclink["sdate"])
    iclink["edate"] = pd.to_datetime(iclink["edate"])

    resolved = _resolve_pit_link(act, iclink, "anndats", "ticker", "sdate", "edate", "permno")
    resolved["permno"] = resolved["permno"].astype(int)
    resolved["fpedats"] = pd.to_datetime(resolved["fpedats"])
    return resolved[["permno", "anndats", "fpedats", "actual"]]


def build_short_interest(raw_shortint: pd.DataFrame, ccm_link: pd.DataFrame) -> pd.DataFrame:
    si = raw_shortint.copy()
    si["settlement_date"] = pd.to_datetime(si["settlement_date"])
    ccm = _prep_ccm_link(ccm_link)

    resolved = _resolve_pit_link(si, ccm, "settlement_date", "gvkey", "linkdt", "linkenddt", "permno")
    resolved["permno"] = resolved["permno"].astype(int)
    return resolved[["permno", "settlement_date", "shortint"]]


def _prep_ccm_link(ccm_link: pd.DataFrame) -> pd.DataFrame:
    """Standard CCM merge convention: primary, research-quality links
    only (LINKTYPE in LU/LC, LINKPRIM in P/C) -- see module docstring
    source #5. A blank LINKENDDT means the link is still active."""
    ccm = ccm_link[
        ccm_link["linktype"].isin(["LU", "LC"]) & ccm_link["linkprim"].isin(["P", "C"])
    ].copy()
    ccm["linkdt"] = pd.to_datetime(ccm["linkdt"])
    ccm["linkenddt"] = pd.to_datetime(ccm["linkenddt"])
    return ccm


def build_sector_crosswalk(company: pd.DataFrame, ccm_link: pd.DataFrame) -> pd.DataFrame:
    """One row per PERMNO per CCM-link validity spell, carrying GSECTOR
    -- consumed directly by features.attach_sector's own merge_asof, so
    this just resolves GVKEY->GSECTOR and GVKEY->PERMNO(+dates) without
    collapsing anything by date itself."""
    ccm = _prep_ccm_link(ccm_link)
    merged = ccm.merge(company, on="gvkey", how="left")
    merged["permno"] = merged["permno"].astype(int)
    return merged[["permno", "linkdt", "linkenddt", "gsector"]].sort_values(["permno", "linkdt"])


def build_siccd_history(raw_siccd: pd.DataFrame) -> pd.DataFrame:
    hist = raw_siccd.copy()
    hist["sic_start"] = pd.to_datetime(hist["sic_start"])
    hist["sic_end"] = pd.to_datetime(hist["sic_end"])
    hist["permno"] = hist["permno"].astype(int)
    return hist[["permno", "sic_start", "sic_end", "siccd"]].sort_values(["permno", "sic_start"])


def _match_rate(df: pd.DataFrame, membership: pd.DataFrame) -> float:
    """Quick sanity signal printed per source: what fraction of PERMNOs
    that were ever S&P 500 members also show up in this source. A very
    low number here usually means the link table join broke, not that
    coverage is genuinely that sparse -- flagged by the instructions'
    own validation checklist item 1."""
    universe = set(membership["permno"].unique())
    matched = set(df["permno"].unique()) & universe
    return len(matched) / len(universe) if universe else float("nan")


def main():
    if not RAW_DIR.exists():
        print(f"Put your WRDS web-query CSV exports in {RAW_DIR} first -- see this "
              f"module's docstring for the 7 expected exports and their filenames.")
        sys.exit(1)

    membership_path = DATA_DIR / "sp500_membership.parquet"
    membership = pd.read_parquet(membership_path) if membership_path.exists() else None

    DATA_DIR.mkdir(exist_ok=True)
    built = {}

    def _try(name, fn):
        try:
            df = fn()
        except FileNotFoundError as e:
            print(f"[skip] {name}: {e}")
            return
        out_path = DATA_DIR / f"{name}.parquet"
        df.to_parquet(out_path, index=False)
        built[name] = df
        match_note = f", {_match_rate(df, membership):.1%} PERMNO match rate" if membership is not None else ""
        print(f"  {len(df)} rows{match_note} -> {out_path}")

    print("Building ibes_estimates...")
    _try("ibes_estimates", lambda: build_ibes_estimates(
        _load("*ibes_summary*.csv", _IBES_SUMMARY_COLS),
        _load("*ibes_crsp_link*.csv", _IBES_LINK_COLS),
    ))

    print("Building ibes_actuals...")
    _try("ibes_actuals", lambda: build_ibes_actuals(
        _load("*ibes_actuals*.csv", _IBES_ACTUALS_COLS),
        _load("*ibes_crsp_link*.csv", _IBES_LINK_COLS),
    ))

    print("Building short_interest...")
    _try("short_interest", lambda: build_short_interest(
        _load("*short_interest*.csv", _SHORTINT_COLS),
        _load("*ccm_link*.csv", _CCM_LINK_COLS),
    ))

    print("Building sector_crosswalk...")
    _try("sector_crosswalk", lambda: build_sector_crosswalk(
        _load("*compustat_company*.csv", _COMPANY_COLS),
        _load("*ccm_link*.csv", _CCM_LINK_COLS),
    ))

    print("Building siccd_history...")
    _try("siccd_history", lambda: build_siccd_history(
        _load("*siccd_history*.csv", _SICCD_COLS)
    ))

    missing = {"ibes_estimates", "ibes_actuals", "short_interest",
               "sector_crosswalk", "siccd_history"} - built.keys()
    if missing:
        print(f"\nNot built (raw export missing): {sorted(missing)}. "
              f"build_feature_panel requires all 5 -- see main.py wiring.")
    else:
        print("\nAll 5 fundamental sources built.")


if __name__ == "__main__":
    main()
