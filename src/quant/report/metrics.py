from __future__ import annotations
import numpy as np
import pandas as pd

def cagr(equity: pd.Series) -> float:
    if len(equity) < 2:
        return 0.0
    total = equity.iloc[-1] / equity.iloc[0]
    years = len(equity) / 252.0
    return float(total ** (1.0 / years) - 1.0)

def max_drawdown(equity: pd.Series) -> float:
    peak = equity.cummax()
    dd = equity / peak - 1.0
    return float(dd.min())

def calmar(equity: pd.Series) -> float:
    dd = abs(max_drawdown(equity))
    if dd == 0:
        return np.inf
    return cagr(equity) / dd

def turnover(weights: pd.DataFrame) -> float:
    # average daily one-way turnover
    dw = weights.diff().abs().sum(axis=1) / 2.0
    return float(dw.mean())

def yearly_returns(equity: pd.Series) -> pd.Series:
    yr = equity.resample("Y").last().pct_change().dropna()
    yr.index = yr.index.year
    return yr
