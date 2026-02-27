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
