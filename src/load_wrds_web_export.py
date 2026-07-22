"""
load_wrds_web_export.py

Alternative to data_prep.py for when the `wrds` Python package can't connect
directly (VPN/driver issues, etc.). Builds the same two cached parquet files
(`data/sp500_membership.parquet`, `data/prices_wrds.parquet`) from a CSV
exported manually via the WRDS website's web-based query tool, so the rest
of the pipeline (features.py, main.py) needs no changes -- this is still
real point-in-time WRDS/CRSP data, just pulled through the browser instead
of the wrds Python API.

Expected input: one or more CSVs in data/raw/ matching *membership_prices*.csv,
in CRSP's newer "CIZ" combined format (index membership pre-joined with the
daily security file, one row per PERMNO-date) -- e.g. from the WRDS web
query "S&P 500 Index Constituents" with the daily price fields included.
Relevant columns: PERMNO, MbrStartDt, MbrEndDt, DlyCalDt, DlyPrc, DlyRet,
DlyVol, ShrOut, DlyCumFacPr, DlyCumFacShr (case handled as-is from the
WRDS export; see _CIZ_COLS below for the exact mapping).

Usage:
    python src/load_wrds_web_export.py
"""

import sys
from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
RAW_DIR = DATA_DIR / "raw"

# WRDS CIZ-format column -> pipeline column name
_CIZ_COLS = {
    "PERMNO": "permno",
    "MbrStartDt": "start",
    "MbrEndDt": "ending",
    "DlyCalDt": "date",
    "DlyPrc": "prc",
    "DlyRet": "ret",
    "DlyVol": "vol",
    "ShrOut": "shrout",
    "DlyCumFacPr": "cfacpr",
    "DlyCumFacShr": "cfacshr",
}


def _load_combined(raw_dir: Path = RAW_DIR) -> pd.DataFrame:
    files = sorted(raw_dir.glob("*membership_prices*.csv"))
    if not files:
        raise FileNotFoundError(
            f"No *membership_prices*.csv found in {raw_dir} -- expected the "
            f"combined CIZ-format WRDS export (membership + daily prices)."
        )

    frames = []
    for f in files:
        df = pd.read_csv(f, usecols=list(_CIZ_COLS.keys()), low_memory=False)
        df = df.rename(columns=_CIZ_COLS)
        frames.append(df)
        print(f"  loaded {f.name} ({len(df)} rows)")

    return pd.concat(frames, ignore_index=True)


def build_membership(combined: pd.DataFrame) -> pd.DataFrame:
    membership = combined[["permno", "start", "ending"]].drop_duplicates().copy()
    membership["permno"] = membership["permno"].astype(int)
    membership["start"] = pd.to_datetime(membership["start"])
    membership["ending"] = pd.to_datetime(membership["ending"])
    return membership.sort_values(["permno", "start"]).reset_index(drop=True)


def build_prices(combined: pd.DataFrame) -> pd.DataFrame:
    cols = ["permno", "date", "prc", "ret", "vol", "shrout", "cfacpr", "cfacshr"]
    panel = combined[cols].copy()
    panel["date"] = pd.to_datetime(panel["date"])

    for col in ["prc", "ret", "vol", "shrout", "cfacpr", "cfacshr"]:
        panel[col] = pd.to_numeric(panel[col], errors="coerce")

    panel = panel.drop_duplicates(subset=["permno", "date"]).sort_values(["permno", "date"])

    # Same CRSP quirk as data_prep.download_prices_wrds -- prc can be
    # negative when it's a bid/ask midpoint estimate rather than an actual
    # trade price; always abs() before using it for price levels.
    panel["prc"] = panel["prc"].abs()
    panel["mkt_cap"] = panel["prc"] * panel["shrout"] * 1000  # shrout is in thousands
    panel["dollar_vol"] = panel["prc"] * panel["vol"]

    return panel.reset_index(drop=True)


def main():
    if not RAW_DIR.exists() or not any(RAW_DIR.glob("*membership_prices*.csv")):
        print(f"Put your WRDS web-query CSV export in {RAW_DIR} first:")
        print("  data/raw/membership_prices.csv   (combined CIZ-format membership + daily prices)")
        sys.exit(1)

    print("Loading combined WRDS web export...")
    combined = _load_combined()

    print("Building membership...")
    membership = build_membership(combined)
    DATA_DIR.mkdir(exist_ok=True)
    membership.to_parquet(DATA_DIR / "sp500_membership.parquet", index=False)
    print(
        f"  {membership['permno'].nunique()} unique PERMNOs, "
        f"{len(membership)} membership spells -> data/sp500_membership.parquet"
    )

    print("Building daily price panel...")
    prices = build_prices(combined)
    prices.to_parquet(DATA_DIR / "prices_wrds.parquet", index=False)
    print(
        f"  {len(prices)} rows, {prices['date'].min().date()} to "
        f"{prices['date'].max().date()} -> data/prices_wrds.parquet"
    )


if __name__ == "__main__":
    main()
