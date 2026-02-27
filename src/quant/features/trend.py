from __future__ import annotations
import pandas as pd

def sma(prices: pd.Series, window: int) -> pd.Series:
    return prices.rolling(window).mean()

def above_ma(prices: pd.Series, window: int) -> pd.Series:
    return prices > sma(prices, window)
