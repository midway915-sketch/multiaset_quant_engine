from __future__ import annotations
from typing import Dict, Tuple, Optional
import pandas as pd
import numpy as np

from quant.backtest.costs import rebalance_cost
from quant.data.return_provider import ReturnProvider


def _price_return(prices: pd.DataFrame, entry_dt: pd.Timestamp, exit_dt: pd.Timestamp, ticker: Optional[str]) -> float:
    if ticker is None:
        return float("nan")
    if ticker not in prices.columns:
        return float("nan")
    if entry_dt not in prices.index or exit_dt not in prices.index:
        return float("nan")
    p0 = prices.at[entry_dt, ticker]
    p1 = prices.at[exit_dt, ticker]
    if not np.isfinite(p0) or p0 == 0 or not np.isfinite(p1):
        return float("nan")
    return float(p1 / p0 - 1.0)


def run_backtest(
    prices: pd.DataFrame,
    signals: pd.DataFrame,
    rp: ReturnProvider,
    costs: Dict[str, float],
) -> Tuple[pd.Series, pd.DataFrame, pd.DataFrame]:
    """Runs daily backtest given precomputed signals.
    Returns: equity, daily_weights, trades

    trades will include:
      - date (apply_date)
      - signal_date
      - assets / weights / gears
      - rank1_ticker / rank1_score / rank2_ticker / rank2_score (if provided by signals)
      - exit_date (next rebalance apply_date, or last price date)
      - rank1_ret / rank2_ret (close-to-close return from date -> exit_date)
    """
    dates = prices.index

    sig_by_apply = {pd.Timestamp(r["apply_date"]): r for _, r in signals.iterrows()}

    equity = []
    eq = 1.0
    daily_w = []
    trades = []

    cur_w: Dict[str, float] = {}
    cur_gears: Dict[str, str] = {}

    for dt in dates:
        if dt in sig_by_apply:
            row = sig_by_apply[dt]
            new_w = row["weights"]
            new_gears = row["gears"]

            c = rebalance_cost(cur_w, new_w, costs["buy"], costs["sell"])

            trade_row = {
                "date": dt,
                "signal_date": row["signal_date"],
                "assets": row["assets"],
                "weights": new_w,
                "gears": new_gears,
                "turnover_cost": c,
            }

            for k in ["rank1_ticker", "rank1_score", "rank2_ticker", "rank2_score"]:
                if k in row:
                    trade_row[k] = row[k]

            trades.append(trade_row)

            cur_w = dict(new_w)
            cur_gears = dict(new_gears)
        else:
            c = 0.0

        if len(cur_w) == 0:
            port_r = 0.0
        else:
            rets = rp.get_returns_matrix(dt, cur_gears)
            port_r = 0.0
            for t, w in cur_w.items():
                port_r += w * rets.get(t, 0.0)

        port_r -= c
        eq *= (1.0 + port_r)
        equity.append(eq)
        daily_w.append(cur_w | {"__date__": dt})

    equity_s = pd.Series(equity, index=dates, name="equity")

    wdf = pd.DataFrame(daily_w).set_index("__date__").fillna(0.0)
    wdf.index.name = "date"

    tdf = pd.DataFrame(trades)
    if len(tdf) > 0:
        tdf["date"] = pd.to_datetime(tdf["date"])
        tdf = tdf.sort_values("date").reset_index(drop=True)

        exit_dates = []
        r1_rets = []
        r2_rets = []
        for i in range(len(tdf)):
            entry_dt = pd.Timestamp(tdf.at[i, "date"])
            if i + 1 < len(tdf):
                exit_dt = pd.Timestamp(tdf.at[i + 1, "date"])
            else:
                exit_dt = pd.Timestamp(dates[-1])

            exit_dates.append(exit_dt)

            r1 = _price_return(prices, entry_dt, exit_dt, tdf.at[i, "rank1_ticker"] if "rank1_ticker" in tdf.columns else None)
            r2 = _price_return(prices, entry_dt, exit_dt, tdf.at[i, "rank2_ticker"] if "rank2_ticker" in tdf.columns else None)
            r1_rets.append(r1)
            r2_rets.append(r2)

        tdf["exit_date"] = exit_dates
        if "rank1_ticker" in tdf.columns:
            tdf["rank1_ret"] = r1_rets
        if "rank2_ticker" in tdf.columns:
            tdf["rank2_ret"] = r2_rets

    return equity_s, wdf, tdf