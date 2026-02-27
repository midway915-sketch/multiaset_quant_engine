from __future__ import annotations
import pandas as pd
from pathlib import Path

def load_prices_long(path: str) -> pd.DataFrame:
    p = Path(path)
    if p.suffix.lower() in [".csv"]:
        df = pd.read_csv(p)
    elif p.suffix.lower() in [".parquet"]:
        df = pd.read_parquet(p)
    else:
        raise ValueError("prices file must be .csv or .parquet")

    if "date" not in df.columns or "ticker" not in df.columns:
        raise ValueError("prices must contain columns: date, ticker, adj_close")
    if "adj_close" not in df.columns:
        raise ValueError("prices must include adj_close")

    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["date", "ticker"])
    return df

def to_wide_adj_close(df_long: pd.DataFrame) -> pd.DataFrame:
    wide = df_long.pivot(index="date", columns="ticker", values="adj_close").sort_index()
    return wide
