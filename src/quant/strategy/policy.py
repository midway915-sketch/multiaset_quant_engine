from __future__ import annotations
from typing import Dict, Optional, Tuple
import pandas as pd
import yaml

from quant.core.calendar import month_end_dates, week_end_dates, next_trading_day
from quant.strategy.scoring import compute_features
from quant.strategy.allocator import choose_top_assets, pick_gear_for_asset


def load_params(path: str) -> Dict:
    return yaml.safe_load(open(path, "r", encoding="utf-8"))


def _top2_from_candidates(candidates: pd.Series) -> Tuple[Optional[str], Optional[float], Optional[str], Optional[float]]:
    if candidates is None or len(candidates) == 0:
        return None, None, None, None
    c = candidates.replace([float("inf"), float("-inf")], pd.NA).dropna().sort_values(ascending=False)
    if len(c) == 0:
        return None, None, None, None
    t1 = str(c.index[0]); s1 = float(c.iloc[0])
    if len(c) >= 2:
        t2 = str(c.index[1]); s2 = float(c.iloc[1])
    else:
        t2 = None; s2 = None
    return t1, s1, t2, s2


def build_signals(
    prices: pd.DataFrame,
    universe_kind_map: Dict[str, str],
    regime_ticker: str,
    cash_proxy: str,
    params: Dict,
) -> pd.DataFrame:
    """
    Output columns:
      signal_date, apply_date, risk_on, assets, weights, gears,
      rank1_ticker, rank1_score, rank2_ticker, rank2_score
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

    slow = int(params["filters"].get("trend_slow", 200))
    feats["_ma_slow"] = prices.rolling(slow).mean()
    feats["_ma200"] = prices.rolling(200).mean()

    regime_ma_days = int(params["filters"]["regime_ma_days"])
    reg = prices[regime_ticker]
    reg_ma = reg.rolling(regime_ma_days).mean()
    regime_on = (reg > reg_ma)

    rows = []
    for dt in rebal_dates:
        if dt not in prices.index:
            continue

        apply_dt = next_trading_day(dates, dt)

        # Risk OFF
        if not bool(regime_on.loc[dt]):
            defensive_kinds = {"bond", "alt", "fx", "alt_trend", "multi_asset"}
            defensive_assets = [a for a, k in universe_kind_map.items() if k in defensive_kinds]
            defensive_assets = [a for a in defensive_assets if a != cash_proxy]

            if len(defensive_assets) == 0:
                rows.append({
                    "signal_date": dt,
                    "apply_date": apply_dt,
                    "risk_on": False,
                    "assets": [cash_proxy],
                    "weights": {cash_proxy: 1.0},
                    "gears": {cash_proxy: "1x"},
                    "rank1_ticker": cash_proxy,
                    "rank1_score": None,
                    "rank2_ticker": None,
                    "rank2_score": None,
                })
                continue

            risk_adj_row = feats["risk_adj"].loc[dt].reindex(defensive_assets)
            candidates = risk_adj_row.replace([float("inf"), float("-inf")], pd.NA).dropna().sort_values(ascending=False)
            r1_t, r1_s, r2_t, r2_s = _top2_from_candidates(candidates)

            if len(candidates) == 0:
                rows.append({
                    "signal_date": dt,
                    "apply_date": apply_dt,
                    "risk_on": False,
                    "assets": [cash_proxy],
                    "weights": {cash_proxy: 1.0},
                    "gears": {cash_proxy: "1x"},
                    "rank1_ticker": cash_proxy,
                    "rank1_score": None,
                    "rank2_ticker": None,
                    "rank2_score": None,
                })
                continue

            chosen = [str(candidates.index[0])]
            rows.append({
                "signal_date": dt,
                "apply_date": apply_dt,
                "risk_on": False,
                "assets": chosen,
                "weights": {chosen[0]: 1.0},
                "gears": {chosen[0]: "1x"},
                "rank1_ticker": r1_t,
                "rank1_score": r1_s,
                "rank2_ticker": r2_t,
                "rank2_score": r2_s,
            })
            continue

        # Risk ON
        pr = prices.loc[dt]
        chosen, wts, ranked_top = choose_top_assets(
            dt,
            pr,
            feats,
            universe_kind_map,
            {"cash_proxy": cash_proxy, "regime_ticker": regime_ticker},
            params,
            top_k=2,
        )

        r1_t = ranked_top[0][0] if len(ranked_top) >= 1 else None
        r1_s = ranked_top[0][1] if len(ranked_top) >= 1 else None
        r2_t = ranked_top[1][0] if len(ranked_top) >= 2 else None
        r2_s = ranked_top[1][1] if len(ranked_top) >= 2 else None

        if len(chosen) == 0:
            rows.append({
                "signal_date": dt,
                "apply_date": apply_dt,
                "risk_on": True,
                "assets": [cash_proxy],
                "weights": {cash_proxy: 1.0},
                "gears": {cash_proxy: "1x"},
                "rank1_ticker": cash_proxy,
                "rank1_score": None,
                "rank2_ticker": None,
                "rank2_score": None,
            })
            continue

        gears = {}
        for a in chosen:
            gears[a] = pick_gear_for_asset(
                dt,
                a,
                universe_kind_map.get(a, "other"),
                prices,
                feats,
                params,
                True,
            )

        rows.append({
            "signal_date": dt,
            "apply_date": apply_dt,
            "risk_on": True,
            "assets": chosen,
            "weights": wts,
            "gears": gears,
            "rank1_ticker": r1_t,
            "rank1_score": r1_s,
            "rank2_ticker": r2_t,
            "rank2_score": r2_s,
        })

    return pd.DataFrame(rows)


def build_monthly_signals(
    prices: pd.DataFrame,
    universe_kind_map: Dict[str, str],
    regime_ticker: str,
    cash_proxy: str,
    params: Dict,
) -> pd.DataFrame:
    return build_signals(prices, universe_kind_map, regime_ticker, cash_proxy, params)