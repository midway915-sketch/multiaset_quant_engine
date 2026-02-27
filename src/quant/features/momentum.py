from __future__ import annotations
import pandas as pd

def pct_change(prices: pd.DataFrame, days: int) -> pd.DataFrame:
    return prices.pct_change(days)
