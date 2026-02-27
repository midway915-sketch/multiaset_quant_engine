from __future__ import annotations
from typing import Dict, List, Tuple
import pandas as pd
import numpy as np

from quant.backtest.costs import rebalance_cost
from quant.data.return_provider import ReturnProvider

def run_backtest(
    prices: pd.DataFrame,
    signals: pd.DataFrame,
    rp: ReturnProvider,
    costs: Dict[str, float],
) -> Tuple[pd.Series, pd.DataFrame, pd.DataFrame]:
    """Runs daily backtest given precomputed monthly signals.
    Returns: equity, daily_weights, trades
    """
    dates = prices.index
    # Build a map apply_date -> signal row
    sig_by_apply = {pd.Timestamp(r["apply_date"]): r for _, r in signals.iterrows()}

    equity = []
    eq = 1.0
    daily_w = []
    trades = []

    cur_w: Dict[str, float] = {}
    cur_gears: Dict[str, str] = {}

    for i, dt in enumerate(dates):
        # Apply rebalance if signal starts today
        if dt in sig_by_apply:
            row = sig_by_apply[dt]
            new_w = row["weights"]
            new_gears = row["gears"]

            c = rebalance_cost(cur_w, new_w, costs["buy"], costs["sell"])
            trades.append({
                "date": dt,
                "signal_date": row["signal_date"],
                "assets": row["assets"],
                "weights": new_w,
                "gears": new_gears,
                "turnover_cost": c,
            })
            cur_w = dict(new_w)
            cur_gears = dict(new_gears)
        else:
            c = 0.0

        # Compute portfolio return using current weights and gears on dt
        if len(cur_w) == 0:
            port_r = 0.0
        else:
            # holdings are base tickers with gear; return provider will use real lever ETF if available
            rets = rp.get_returns_matrix(dt, cur_gears)
            port_r = 0.0
            for t, w in cur_w.items():
                port_r += w * rets.get(t, 0.0)

        port_r -= c
        eq *= (1.0 + port_r)
        equity.append(eq)
        daily_w.append(cur_w | {"__date__": dt})

    equity_s = pd.Series(equity, index=dates, name="equity")

    # weights df
    wdf = pd.DataFrame(daily_w).set_index("__date__").fillna(0.0)
    wdf.index.name = "date"

    tdf = pd.DataFrame(trades)
    if len(tdf) > 0:
        tdf["date"] = pd.to_datetime(tdf["date"])
        tdf = tdf.sort_values("date")

    return equity_s, wdf, tdf
