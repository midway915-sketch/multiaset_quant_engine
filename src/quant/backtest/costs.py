from __future__ import annotations
from typing import Dict, Tuple

def rebalance_cost(prev_w: Dict[str, float], new_w: Dict[str, float], buy: float, sell: float) -> float:
    """Simple cost model:
    - cost on decreases (sell) and increases (buy) in weights (absolute change).
    - returns a *portfolio return* drag (subtract from return that day).
    """
    tickers = set(prev_w) | set(new_w)
    cost = 0.0
    for t in tickers:
        p = prev_w.get(t, 0.0)
        n = new_w.get(t, 0.0)
        if n > p:
            cost += (n - p) * buy
        elif p > n:
            cost += (p - n) * sell
    return cost
