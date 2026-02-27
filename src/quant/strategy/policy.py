from __future__ import annotations
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
import yaml

from quant.core.calendar import month_end_dates, next_trading_day
from quant.features.trend import sma
from quant.strategy.scoring import compute_features
from quant.strategy.allocator import choose_top_assets, pick_gear_for_asset

def load_params(path: str) -> Dict:
    return yaml.safe_load(open(path, "r", encoding="utf-8"))

def build_monthly_signals(
    prices: pd.DataFrame,
    universe_kind_map: Dict[str, str],
    regime_ticker: str,
    cash_proxy: str,
    params: Dict,
) -> pd.DataFrame:
    dates = prices.index
    rebal_dates = month_end_dates(dates)

    feats = compute_features(prices, params)

    # MA200 for absolute filter: computed per-asset
    slow = params["filters"]["trend_slow"]
    ma200_all = prices.rolling(slow).mean()
    feats["_ma200"] = ma200_all

    # regime
    regime_ma_days = params["filters"]["regime_ma_days"]
    spy = prices[regime_ticker]
    spy_ma = spy.rolling(regime_ma_days).mean()
    regime_on = (spy > spy_ma)

    rows = []
    for dt in rebal_dates:
        if dt not in prices.index:
            continue
        # signal computed at dt; applied from next trading day
        apply_dt = next_trading_day(dates, dt)

        if not bool(regime_on.loc[dt]):
            rows.append({
                "signal_date": dt,
                "apply_date": apply_dt,
                "risk_on": False,
                "assets": [cash_proxy],
                "weights": {cash_proxy: 1.0},
                "gears": {cash_proxy: "1x"},
            })
            continue

        pr = prices.loc[dt]
        chosen, wts = choose_top_assets(dt, pr, feats, universe_kind_map, {}, params)
        if len(chosen) == 0:
            # fallback to cash if no candidates
            rows.append({
                "signal_date": dt,
                "apply_date": apply_dt,
                "risk_on": True,
                "assets": [cash_proxy],
                "weights": {cash_proxy: 1.0},
                "gears": {cash_proxy: "1x"},
            })
            continue

        gears = {}
        for a in chosen:
            gears[a] = pick_gear_for_asset(dt, a, universe_kind_map.get(a, "other"), prices, feats, params, True)

        rows.append({
            "signal_date": dt,
            "apply_date": apply_dt,
            "risk_on": True,
            "assets": chosen,
            "weights": wts,
            "gears": gears,
        })

    df = pd.DataFrame(rows)
    return df
