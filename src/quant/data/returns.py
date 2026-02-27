from __future__ import annotations
import numpy as np
import pandas as pd

def pct_returns(prices: pd.DataFrame) -> pd.DataFrame:
    return prices.pct_change()

def log_returns(prices: pd.DataFrame) -> pd.DataFrame:
    return np.log(prices / prices.shift(1))

def apply_daily_fee(r: pd.Series | pd.DataFrame, annual_fee: float) -> pd.Series | pd.DataFrame:
    return r - (annual_fee / 252.0)

def synthetic_leverage(r_base: pd.Series, multiple: float) -> pd.Series:
    return multiple * r_base
