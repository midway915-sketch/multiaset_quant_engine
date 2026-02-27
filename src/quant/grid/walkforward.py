from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Tuple
import pandas as pd

@dataclass(frozen=True)
class WFWindow:
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp

def make_walkforward_windows(dates: pd.DatetimeIndex, train_years: int, test_years: int, step_years: int) -> List[WFWindow]:
    start = dates.min()
    end = dates.max()
    windows = []
    cur = pd.Timestamp(start.year, 1, 1)
    while True:
        train_start = cur
        train_end = pd.Timestamp(cur.year + train_years, 1, 1) - pd.Timedelta(days=1)
        test_start = pd.Timestamp(cur.year + train_years, 1, 1)
        test_end = pd.Timestamp(cur.year + train_years + test_years, 1, 1) - pd.Timedelta(days=1)
        if test_end > end:
            break
        windows.append(WFWindow(train_start, train_end, test_start, test_end))
        cur = pd.Timestamp(cur.year + step_years, 1, 1)
    return windows
