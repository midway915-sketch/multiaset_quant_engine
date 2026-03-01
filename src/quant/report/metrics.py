from __future__ import annotations
import numpy as np
import pandas as pd

def _slice_last_years(equity: pd.Series, years: int) -> pd.Series:
    """Return equity restricted to the last N years (calendar years back from the last timestamp).

    If there isn't enough history, returns the original series.
    """
    if equity is None or len(equity) < 2:
        return equity
    # Ensure datetime index
    if not isinstance(equity.index, pd.DatetimeIndex):
        equity = equity.copy()
        equity.index = pd.to_datetime(equity.index)

    end = equity.index.max()
    start = end - pd.DateOffset(years=years)
    sliced = equity.loc[equity.index >= start]
    return sliced if len(sliced) >= 2 else equity

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

def last_n_years_metrics(equity: pd.Series, years: int = 10) -> dict:
    """Convenience metrics over the last N years window.

    Returns:
      - start/end dates
      - seed multiple (end/start)
      - CAGR
      - MDD
    """
    eq = _slice_last_years(equity, years)
    if eq is None or len(eq) < 2:
        return {
            "start": None,
            "end": None,
            "multiple": 1.0,
            "cagr": 0.0,
            "mdd": 0.0,
        }

    start = str(eq.index.min().date())
    end = str(eq.index.max().date())
    multiple = float(eq.iloc[-1] / eq.iloc[0])
    return {
        "start": start,
        "end": end,
        "multiple": multiple,
        "cagr": cagr(eq),
        "mdd": max_drawdown(eq),
    }