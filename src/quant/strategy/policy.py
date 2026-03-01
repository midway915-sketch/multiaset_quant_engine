from __future__ import annotations
from typing import Dict, Optional, Tuple, List
import pandas as pd
import yaml

from quant.core.calendar import month_end_dates, week_end_dates, next_trading_day
from quant.strategy.scoring import compute_features
from quant.strategy.allocator import choose_top_assets, pick_gear_for_asset, risk_on_candidates


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


def _mean_score(candidates: pd.Series, assets: List[str]) -> Optional[float]:
    if candidates is None or len(candidates) == 0:
        return None
    if not assets:
        return None
    vals = []
    for a in assets:
        if a in candidates.index:
            vals.append(float(candidates.loc[a]))
        else:
            return None  # if any asset not eligible now, cannot keep
    if len(vals) == 0:
        return None
    return float(sum(vals) / len(vals))


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

    # --- hysteresis config ---
    sel = params.get("selection", {}) or {}
    hcfg = sel.get("hysteresis", {}) or {}
    hyst_enabled = bool(hcfg.get("enabled", False))
    # score improvement threshold: only switch if new portfolio mean score improves by at least this amount
    hyst_min_improve = float(hcfg.get("min_improve", 0.0))

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
            # --- HYSTERESIS (Risk OFF) ---
            # prev 보유 방어자산이 현재 후보군에 남아있고,
            # 새 1등 점수 개선이 min_improve보다 작으면 교체하지 않음.
            if hyst_enabled and len(rows) > 0:
                prev = rows[-1]
                if bool(prev.get("risk_on", True)) is False:
                    prev_assets = prev.get("assets", []) or []
                    if len(prev_assets) >= 1:
                        prev_a = str(prev_assets[0])
                        if prev_a in candidates.index and len(candidates) >= 1:
                            prev_score = float(candidates.loc[prev_a])
                            best_score = float(candidates.iloc[0])
                            if (best_score - prev_score) < hyst_min_improve:
                                # keep previous defensive holding
                                chosen = [prev_a]
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

        # compute full candidates (after filters) so we can apply hysteresis against previous holdings
        candidates = risk_on_candidates(
            dt,
            pr,
            feats,
            {"cash_proxy": cash_proxy, "regime_ticker": regime_ticker},
            params,
        )

        # standard choice (top_n equal weights)
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

        # --- HYSTERESIS: keep previous holdings if improvement is too small ---
        if hyst_enabled and len(rows) > 0:
            prev = rows[-1]
            # only apply when previous was also risk_on (avoid sticking across regime flip)
            if bool(prev.get("risk_on", False)) is True:
                prev_assets = prev.get("assets", []) or []
                # Compare mean score of previous holdings vs mean score of new chosen (using CURRENT candidates)
                prev_mean = _mean_score(candidates, [str(a) for a in prev_assets])
                new_mean = _mean_score(candidates, [str(a) for a in chosen])

                # if previous holdings are still eligible now, and improvement is small -> keep prev
                if prev_mean is not None and new_mean is not None:
                    if (new_mean - prev_mean) < hyst_min_improve:
                        chosen = [str(a) for a in prev_assets]
                        if len(chosen) == 1:
                            wts = {chosen[0]: 1.0}
                        else:
                            w = 1.0 / len(chosen)
                            wts = {t: w for t in chosen}

        # compute gears for actually held assets
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
            # keep ranking info for diagnostics (even if we decided not to switch)
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