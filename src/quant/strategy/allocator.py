from __future__ import annotations
from typing import Dict, List, Tuple, Optional
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
    top_k: int = 2,
) -> Tuple[List[str], Dict[str, float], List[Tuple[str, float]]]:
    """
    Choose top assets under absolute filters and return:

      - chosen: list of assets to actually hold (size = params['selection']['top_n'])
      - weights: dict of weights for chosen assets
      - ranked_top: list of (ticker, score) for the top_k candidates (for logging),
                    where score is feats['risk_adj'] for that dt.
    """
    top_n = int(params["selection"]["top_n"])
    abs_mom_min = float(params["filters"]["abs_mom_min"])

    risk_adj_row = feats["risk_adj"].loc[dt]
    abs_mom_row = feats["abs_mom"].loc[dt]

    # --- robust alignment ---
    risk_adj_row = risk_adj_row.reindex(prices_row.index)
    abs_mom_row = abs_mom_row.reindex(prices_row.index)

    # MA200 (per-asset) injected by policy
    ma200_row = feats.get("_ma200", pd.DataFrame()).loc[dt] if "_ma200" in feats else pd.Series(index=prices_row.index, data=np.nan)
    if isinstance(ma200_row, pd.Series):
        ma200_row = ma200_row.reindex(prices_row.index)
    else:
        ma200_row = pd.Series(index=prices_row.index, data=np.nan)

    # absolute filters per asset
    pass_abs = (abs_mom_row > abs_mom_min) & (prices_row > ma200_row)
    pass_abs = pass_abs.fillna(False)

    # cash proxy should NOT be part of "top assets" ranking (fallback handled elsewhere)
    cash_proxy = (universe_cfg.get("cash_proxy") or params.get("cash_proxy") or "BIL")
    if cash_proxy in pass_abs.index:
        pass_abs.loc[cash_proxy] = False

    candidates = risk_adj_row[pass_abs].replace([np.inf, -np.inf], np.nan).dropna()
    candidates = candidates.sort_values(ascending=False)

    ranked_top = [(str(k), float(v)) for k, v in candidates.head(int(top_k)).items()]

    chosen = candidates.index[:top_n].tolist()
    if len(chosen) == 0:
        return [], {}, ranked_top
    if len(chosen) == 1:
        return chosen, {chosen[0]: 1.0}, ranked_top
    w = 1.0 / len(chosen)
    return chosen, {t: w for t in chosen}, ranked_top


def _realized_vol_annual(prices: pd.Series, dt: pd.Timestamp, lookback: int) -> float:
    """
    Realized annualized vol using daily close-to-close returns over lookback days ending at dt.
    """
    if prices is None or len(prices) == 0:
        return np.nan
    if dt not in prices.index:
        return np.nan
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

    # trend confirm optional (applies before any gear logic)
    if lev.get("use_trend_confirm", True):
        fast = int(params["filters"]["trend_fast"])
        slow = int(params["filters"]["trend_slow"])
        ma_fast = prices[asset].rolling(fast).mean()
        ma_slow = prices[asset].rolling(slow).mean()
        if not (ma_fast.loc[dt] > ma_slow.loc[dt]):
            return "1x"

    # Hybrid Bull-3x mode
    if lev.get("use_hybrid_bull_3x", False):
        bull_ticker = str(lev.get("bull_vol_ticker", "SPY"))
        bull_lookback = int(lev.get("bull_vol_lookback_days", 21))
        bull_vol_max = float(lev.get("bull_vol_max_annual", 0.25))
        force_3x_assets = lev.get("force_3x_assets", ["QQQ", "SMH"])

        if bull_ticker in prices.columns and asset in force_3x_assets:
            mvol = _realized_vol_annual(prices[bull_ticker], dt, bull_lookback)
            if np.isfinite(mvol) and mvol <= bull_vol_max:
                return "3x"

    # Vol targeting leverage mode (discretized to 1x/2x/3x)
    if lev.get("use_vol_target", False):
        target = float(lev.get("vol_target_annual", 0.30))
        lookback = int(lev.get("vol_lookback_days", 63))
        max_lev = float(lev.get("max_leverage", 3.0))

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

    # Existing: ratio mode (v20 / v1y)
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