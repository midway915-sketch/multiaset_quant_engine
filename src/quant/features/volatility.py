from __future__ import annotations
import numpy as np
import pandas as pd

def ann_vol_log(prices: pd.DataFrame, window: int) -> pd.DataFrame:
    logret = np.log(prices / prices.shift(1))
    return logret.rolling(window).std() * (252 ** 0.5)

def vol_ratio(vol_short: pd.Series, vol_ref: pd.Series) -> pd.Series:
    return vol_short / vol_ref.replace(0, np.nan)
