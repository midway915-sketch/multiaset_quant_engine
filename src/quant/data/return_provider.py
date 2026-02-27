from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Optional, Tuple
import pandas as pd

from .returns import pct_returns, apply_daily_fee, synthetic_leverage

@dataclass
class LeverMap:
    base: str
    lev2: Optional[str] = None
    lev3: Optional[str] = None

class ReturnProvider:
    """Provides daily returns for 1x/2x/3x where:
    - If a real leveraged ETF exists (inception passed) and has data for date -> use real ETF return.
    - Else use synthetic leverage from base return.
    Adds daily fee for leveraged gears (configurable).
    """

    def __init__(
        self,
        prices_wide: pd.DataFrame,
        inception: Dict[str, str],
        annual_fees: Dict[str, float],
        leverage_maps: Dict[str, LeverMap],
    ):
        self.prices = prices_wide.copy()
        self.rets = pct_returns(self.prices)
        self.inception = {k: pd.Timestamp(v) for k, v in inception.items()}
        self.annual_fees = annual_fees
        self.maps = leverage_maps

    def _has_real(self, ticker: str, dt: pd.Timestamp) -> bool:
        if ticker not in self.rets.columns:
            return False
        if ticker in self.inception and dt < self.inception[ticker]:
            return False
        val = self.rets.at[dt, ticker] if dt in self.rets.index else None
        return pd.notna(val)

    def get_return(self, base_ticker: str, dt: pd.Timestamp, gear: str) -> float:
        if gear == "1x":
            r = self.rets.at[dt, base_ticker]
            return float(r) if pd.notna(r) else 0.0

        m = self.maps.get(base_ticker)
        if m is None:
            # no mapping; fallback to synthetic
            base_r = self.rets.at[dt, base_ticker]
            base_r = float(base_r) if pd.notna(base_r) else 0.0
            mult = 2.0 if gear == "2x" else 3.0
            r = synthetic_leverage(pd.Series([base_r]), mult).iloc[0]
            r = float(apply_daily_fee(pd.Series([r]), self.annual_fees[gear]).iloc[0])
            return r

        lev_ticker = m.lev2 if gear == "2x" else m.lev3
        mult = 2.0 if gear == "2x" else 3.0

        if lev_ticker and self._has_real(lev_ticker, dt):
            r = self.rets.at[dt, lev_ticker]
            r = float(r) if pd.notna(r) else 0.0
            # Real ETF already includes fee implicitly; we DO NOT add fee again.
            return r

        # synthetic fallback
        base_r = self.rets.at[dt, base_ticker]
        base_r = float(base_r) if pd.notna(base_r) else 0.0
        r = synthetic_leverage(pd.Series([base_r]), mult).iloc[0]
        r = float(apply_daily_fee(pd.Series([r]), self.annual_fees[gear]).iloc[0])
        return r

    def get_returns_matrix(self, dt: pd.Timestamp, holdings: Dict[str, str]) -> Dict[str, float]:
        """holdings: {base_ticker: gear} -> returns"""
        out = {}
        for t, g in holdings.items():
            out[t] = self.get_return(t, dt, g)
        return out
