from __future__ import annotations
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd

from quant.features.trend import sma


def choose_top_assets(
    dt: pd.Timestamp,
    prices_row: pd.Series,
    feats: Dict[str, pd.DataFrame],
    universe_kinds: Dict[str, str],
    universe_cfg: Dict,
    params: Dict,
) -> Tuple[List[str], Dict[str, float]]:
    top_n = params["selection"]["top_n"]
    abs_mom_min = params["filters"]["abs_mom_min"]
    trend_slow = params["filters"]["trend_slow"]

    risk_adj_row = feats["risk_adj"].loc[dt]
    abs_mom_row = feats["abs_mom"].loc[dt]

    # absolute filters per asset
    ma200 = feats["_ma200"].loc[dt]  # injected by policy
    pass_abs = (abs_mom_row > abs_mom_min) & (prices_row > ma200)

    candidates = risk_adj_row[pass_abs].dropna()
    candidates = candidates.sort_values(ascending=False)

    chosen = candidates.index[:top_n].tolist()
    if len(chosen) == 0:
        return [], {}
    if len(chosen) == 1:
        return chosen, {chosen[0]: 1.0}
    # equal weighting
    w = 1.0 / len(chosen)
    return chosen, {t: w for t in chosen}


def _realized_vol_annual(prices: pd.Series, dt: pd.Timestamp, lookback: int) -> float:
    """
    Realized annualized vol using daily close-to-close returns over lookback days ending at dt.
    """
    if prices is None or len(prices) == 0:
        return np.nan
    if dt not in prices.index:
        return np.nan
    # slice window up to dt (inclusive)
    w = prices.loc[:dt].tail(lookback + 1)
    if len(w) < max(lookback // 2, 10):
        return np.nan
    rets = w.pct_change().dropna()
    if len(rets) < max(lookback // 2, 10):
        return np.nan
    vol_daily = float(rets.std(ddof=0))
    if not np.isfinite(vol_daily) or vol_daily <= 0:
        return np.nan
    return vol_daily * np.sqrt(252.0)


def pick_gear_for_asset(
    dt: pd.Timestamp,
    asset: str,
    kind: str,
    prices: pd.DataFrame,
    feats: Dict[str, pd.DataFrame],
    params: Dict,
    regime_on: bool,
) -> str:
    lev = params["leverage"]
    if not lev["enabled"]:
        return "1x"
    if lev.get("equity_only", True) and kind != "equity":
        return "1x"
    if not regime_on:
        return "1x"

    # trend confirm optional
    if lev.get("use_trend_confirm", True):
        fast = params["filters"]["trend_fast"]
        slow = params["filters"]["trend_slow"]
        ma_fast = prices[asset].rolling(fast).mean()
        ma_slow = prices[asset].rolling(slow).mean()
        if not (ma_fast.loc[dt] > ma_slow.loc[dt]):
            return "1x"

    # -----------------------------
    # NEW: Vol targeting leverage mode (discretized to 1x/2x/3x)
    # -----------------------------
    if lev.get("use_vol_target", False):
        target = float(lev.get("vol_target_annual", 0.30))
        lookback = int(lev.get("vol_lookback_days", 63))
        max_lev = float(lev.get("max_leverage", 3.0))

        # cutoffs to map continuous L -> discrete gear
        cut_2x = float(lev.get("gear_cut_2x", 1.5))  # >= this => 2x
        cut_3x = float(lev.get("gear_cut_3x", 2.5))  # >= this => 3x

        vol = _realized_vol_annual(prices[asset], dt, lookback)
        if not np.isfinite(vol) or vol <= 0:
            return "1x"

        L = target / vol
        if not np.isfinite(L):
            return "1x"
        L = max(0.0, min(max_lev, L))

        if L >= cut_3x:
            return "3x"
        if L >= cut_2x:
            return "2x"
        return "1x"

    # -----------------------------
    # Existing: ratio mode (v20 / v1y)
    # -----------------------------
    v20 = feats["vol20"].loc[dt, asset]
    v1y = feats["vol1y"].loc[dt, asset]
    if pd.isna(v20) or pd.isna(v1y) or v1y == 0:
        return "1x"
    ratio = v20 / v1y

    if ratio < lev["thr_3x"]:
        return "3x"
    if ratio < lev["thr_2x"]:
        return "2x"
    return "1x"