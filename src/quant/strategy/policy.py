from __future__ import annotations
from typing import Dict
import pandas as pd
import yaml

from quant.core.calendar import month_end_dates, week_end_dates, next_trading_day
from quant.strategy.scoring import compute_features
from quant.strategy.allocator import choose_top_assets, pick_gear_for_asset

def load_params(path: str) -> Dict:
    return yaml.safe_load(open(path, "r", encoding="utf-8"))

def build_signals(
    prices: pd.DataFrame,
    universe_kind_map: Dict[str, str],
    regime_ticker: str,
    cash_proxy: str,
    params: Dict,
) -> pd.DataFrame:
    """
    Build rebalance signals (monthly/weekly) based on params['rebalance'].

    Output columns:
      signal_date, apply_date, risk_on, assets, weights, gears
    """
    dates = prices.index

    reb = params.get("rebalance", {}) or {}
    frequency = str(reb.get("frequency", "monthly")).lower()
    when = str(reb.get("when", "month_end")).lower()

    # Decide rebalance dates
    if frequency == "weekly" or when == "week_end":
        rebal_dates = week_end_dates(dates)
    else:
        rebal_dates = month_end_dates(dates)

    feats = compute_features(prices, params)

    # MA(trend_slow) per-asset for trend checks
    slow = params["filters"]["trend_slow"]
    ma_slow_all = prices.rolling(slow).mean()
    feats["_ma_slow"] = ma_slow_all

    # regime filter on regime_ticker
    regime_ma_days = params["filters"]["regime_ma_days"]
    reg = prices[regime_ticker]
    reg_ma = reg.rolling(regime_ma_days).mean()
    regime_on = (reg > reg_ma)

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
            gears[a] = pick_gear_for_asset(
                dt, a, universe_kind_map.get(a, "other"),
                prices, feats, params, True
            )

        rows.append({
            "signal_date": dt,
            "apply_date": apply_dt,
            "risk_on": True,
            "assets": chosen,
            "weights": wts,
            "gears": gears,
        })

    return pd.DataFrame(rows)

def build_monthly_signals(
    prices: pd.DataFrame,
    universe_kind_map: Dict[str, str],
    regime_ticker: str,
    cash_proxy: str,
    params: Dict,
) -> pd.DataFrame:
    # Backward compatible wrapper (legacy name)
    return build_signals(prices, universe_kind_map, regime_ticker, cash_proxy, params)