from __future__ import annotations
import numpy as np
import pandas as pd

def _ensure_dt_index(s: pd.Series) -> pd.Series:
    if not isinstance(s.index, pd.DatetimeIndex):
        s = s.copy()
        s.index = pd.to_datetime(s.index)
    return s

def _slice_last_years(equity: pd.Series, years: int) -> pd.Series:
    """Slice equity to the last N years (calendar years back from the last timestamp).
    If history is shorter than N years, returns the full series.
    """
    if equity is None or len(equity) < 2:
        return equity
    eq = _ensure_dt_index(equity)
    end = eq.index.max()
    start = end - pd.DateOffset(years=years)
    sliced = eq.loc[eq.index >= start]
    return sliced if len(sliced) >= 2 else eq

def cagr(equity: pd.Series) -> float:
    if len(equity) < 2:
        return 0.0
    eq = _ensure_dt_index(equity)
    total = eq.iloc[-1] / eq.iloc[0]
    years = len(eq) / 252.0
    return float(total ** (1.0 / years) - 1.0)

def max_drawdown(equity: pd.Series) -> float:
    if len(equity) < 2:
        return 0.0
    eq = _ensure_dt_index(equity)
    peak = eq.cummax()
    dd = eq / peak - 1.0
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
    """
    pandas >= 3.0: 'Y' is removed. Use 'YE' (year-end) instead.
    Keep a fallback for older pandas that may not recognize 'YE'.
    """
    eq = _ensure_dt_index(equity)
    try:
        yr = eq.resample("YE").last().pct_change().dropna()
    except ValueError:
        # Older pandas fallback
        yr = eq.resample("A-DEC").last().pct_change().dropna()

    # Make index just the year integer (e.g., 2024)
    yr.index = yr.index.year
    return yr

def last_n_years_metrics(equity: pd.Series, years: int = 10) -> dict:
    """Exact metrics computed on the last N years slice of the equity curve."""
    if equity is None or len(equity) < 2:
        return {
            "start": None,
            "end": None,
            "multiple": 1.0,
            "cagr": 0.0,
            "mdd": 0.0,
        }

    eq = _slice_last_years(equity, years)
    eq = _ensure_dt_index(eq)

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
