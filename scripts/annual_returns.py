from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd


def read_equity(path: Path) -> pd.Series:
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date")
    s = pd.Series(df["equity"].astype(float).values, index=df["date"])
    s = s[~s.index.duplicated(keep="last")]
    return s.dropna()


def annual_returns(eq: pd.Series) -> pd.Series:
    # calendar year end equity
    ye = eq.resample("YE").last()
    ret = ye / ye.shift(1) - 1.0
    ret = ret.dropna()
    ret.index = ret.index.year
    return ret


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", action="append", required=True,
                    help="format: name=path/to/equity.csv  (repeatable)")
    ap.add_argument("--out", default="annual_returns.csv")
    args = ap.parse_args()

    cols = []
    for item in args.label:
        name, path = item.split("=", 1)
        eq = read_equity(Path(path))
        r = annual_returns(eq).rename(name)
        cols.append(r)

    out = pd.concat(cols, axis=1).sort_index()
    out.to_csv(args.out, float_format="%.6f")
    print(f"saved {args.out}")
    print((out * 100).round(2).to_string())


if __name__ == "__main__":
    main()