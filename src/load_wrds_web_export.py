"""
load_wrds_web_export.py

Alternative to data_prep.py for when the `wrds` Python package can't connect
directly (VPN/driver issues, etc.). Builds the same two cached parquet files
(`data/sp500_membership.parquet`, `data/prices_wrds.parquet`) from CSVs
exported manually via the WRDS website's web-based query tool, so the rest
of the pipeline (features.py, main.py) needs no changes -- this is still
real point-in-time WRDS/CRSP data, just pulled through the browser instead
of the wrds Python API.

Expected inputs (place in data/raw/):
  - One or more membership CSVs from CRSP's S&P 500 index list
    (dsp500list_v2 / dsp500list), with columns permno, mbrstartdt, mbrenddt
    (case-insensitive; common WRDS web-export header variants are handled).
  - One or more daily price CSVs from crsp.dsf, with columns permno, date,
    prc, ret, vol, shrout, cfacpr, cfacshr. The WRDS web query tool caps
    output size, so a full 2012-2026 pull across ~500+ PERMNOs will likely
    need to be split into multiple date-range or PERMNO-batch exports --
    just drop them all in data/raw/, this concatenates every prices*.csv.

I web queried from
https://wrds-www-wharton-upenn-edu.libproxy.mit.edu/pages/get-data/center-research-security-prices-crsp/quarterly-update/index-version-2/sp-500-index-constituents-q/

Usage:
    python src/load_wrds_web_export.py
"""




import sys
from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
RAW_DIR = DATA_DIR / "raw"


def _lower_cols(df: pd.DataFrame) -> pd.DataFrame:
    df.columns = [c.strip().lower() for c in df.columns]
    return df


def _find_col(df: pd.DataFrame, *candidates: str) -> str:
    for c in candidates:
        if c in df.columns:
            return c
    raise KeyError(
        f"None of {candidates} found in columns {list(df.columns)} -- "
        f"check the WRDS export's header names and add the variant to "
        f"_find_col if needed."
    )


def build_membership(raw_dir: Path = RAW_DIR) -> pd.DataFrame:
    files = sorted(raw_dir.glob("membership*.csv"))
    if not files:
        raise FileNotFoundError(f"No membership*.csv found in {raw_dir}")

    frames = []
    for f in files:
        df = _lower_cols(pd.read_csv(f))
        permno_col = _find_col(df, "permno")
        start_col = _find_col(df, "mbrstartdt", "start")
        end_col = _find_col(df, "mbrenddt", "ending", "end")
        frames.append(
            df[[permno_col, start_col, end_col]].rename(
                columns={permno_col: "permno", start_col: "start", end_col: "ending"}
            )
        )

    membership = pd.concat(frames, ignore_index=True)
    membership["permno"] = membership["permno"].astype(int)
    membership["start"] = pd.to_datetime(membership["start"])
    membership["ending"] = pd.to_datetime(membership["ending"])
    membership = membership.drop_duplicates().sort_values(["permno", "start"])
    return membership.reset_index(drop=True)


def build_prices(raw_dir: Path = RAW_DIR) -> pd.DataFrame:
    files = sorted(raw_dir.glob("prices*.csv"))
    if not files:
        raise FileNotFoundError(f"No prices*.csv found in {raw_dir}")

    frames = []
    for f in files:
        df = _lower_cols(pd.read_csv(f))
        col_map = {
            "permno": _find_col(df, "permno"),
            "date": _find_col(df, "date"),
            "prc": _find_col(df, "prc"),
            "ret": _find_col(df, "ret"),
            "vol": _find_col(df, "vol"),
            "shrout": _find_col(df, "shrout"),
            "cfacpr": _find_col(df, "cfacpr"),
            "cfacshr": _find_col(df, "cfacshr"),
        }
        frames.append(df[list(col_map.values())].rename(columns={v: k for k, v in col_map.items()}))
        print(f"  loaded {f.name} ({len(df)} rows)")

    panel = pd.concat(frames, ignore_index=True)
    panel["date"] = pd.to_datetime(panel["date"])

    for col in ["prc", "ret", "vol", "shrout", "cfacpr", "cfacshr"]:
        panel[col] = pd.to_numeric(panel[col], errors="coerce")

    panel = panel.drop_duplicates(subset=["permno", "date"]).sort_values(["permno", "date"])

    # Same CRSP quirks/derivations as data_prep.download_prices_wrds --
    # prc is negative when it's a bid/ask midpoint estimate rather than an
    # actual trade price; always abs() before using it for price levels.
    panel["prc"] = panel["prc"].abs()
    panel["mkt_cap"] = panel["prc"] * panel["shrout"] * 1000  # shrout is in thousands
    panel["dollar_vol"] = panel["prc"] * panel["vol"]

    return panel.reset_index(drop=True)


def main():
    if not RAW_DIR.exists() or not any(RAW_DIR.iterdir()):
        print(f"Put your WRDS web-query CSV exports in {RAW_DIR} first:")
        print("  data/raw/membership.csv        (permno, mbrstartdt, mbrenddt)")
        print("  data/raw/prices_*.csv          (permno, date, prc, ret, vol, shrout, cfacpr, cfacshr)")
        sys.exit(1)

    print("Building membership from WRDS web export...")
    membership = build_membership()
    DATA_DIR.mkdir(exist_ok=True)
    membership.to_parquet(DATA_DIR / "sp500_membership.parquet", index=False)
    print(
        f"  {membership['permno'].nunique()} unique PERMNOs, "
        f"{len(membership)} membership spells -> data/sp500_membership.parquet"
    )

    print("Building daily price panel from WRDS web export...")
    prices = build_prices()
    prices.to_parquet(DATA_DIR / "prices_wrds.parquet", index=False)
    print(
        f"  {len(prices)} rows, {prices['date'].min().date()} to "
        f"{prices['date'].max().date()} -> data/prices_wrds.parquet"
    )


if __name__ == "__main__":
    main()
