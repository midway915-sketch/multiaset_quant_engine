from __future__ import annotations
import pandas as pd

def month_end_dates(trading_dates: pd.DatetimeIndex) -> pd.DatetimeIndex:
    # pick last trading day of each month
    s = pd.Series(trading_dates, index=trading_dates)
    grp = s.groupby([trading_dates.year, trading_dates.month]).max()
    return pd.DatetimeIndex(grp.values).sort_values()

def week_end_dates(trading_dates: pd.DatetimeIndex) -> pd.DatetimeIndex:
    # pick last trading day of each ISO week (Mon-Sun)
    iso = trading_dates.isocalendar()
    s = pd.Series(trading_dates, index=trading_dates)
    grp = s.groupby([iso["year"], iso["week"]]).max()
    return pd.DatetimeIndex(grp.values).sort_values()

def next_trading_day(trading_dates: pd.DatetimeIndex, dt: pd.Timestamp) -> pd.Timestamp:
    i = trading_dates.searchsorted(dt)
    if i < len(trading_dates) and trading_dates[i] == dt:
        i += 1
    if i >= len(trading_dates):
        return trading_dates[-1]
    return trading_dates[i]