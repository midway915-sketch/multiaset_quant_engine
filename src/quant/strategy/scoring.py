from __future__ import annotations
from typing import Dict, Tuple
import numpy as np
import pandas as pd

from quant.features.momentum import pct_change
from quant.features.volatility import ann_vol_log

def compute_features(prices: pd.DataFrame, params: Dict) -> Dict[str, pd.DataFrame]:
    w = params["weights"]
    volcfg = params["volatility"]
    filt = params["filters"]

    ret_1m = pct_change(prices, 21)
    ret_6m = pct_change(prices, 126)
    ret_12m = pct_change(prices, 252)

    score = w["w_1m"] * ret_1m + w["w_6m"] * ret_6m + w["w_12m"] * ret_12m

    risk_vol = ann_vol_log(prices, volcfg["risk_window"])  # e.g., 63d
    risk_adj = score / risk_vol.replace(0, np.nan)

    abs_mom = pct_change(prices, filt["abs_mom_days"])

    vol20 = ann_vol_log(prices, volcfg["vol_window"])
    vol1y = ann_vol_log(prices, volcfg["vol_ref_window"])

    return {
        "score": score,
        "risk_adj": risk_adj,
        "abs_mom": abs_mom,
        "vol20": vol20,
        "vol1y": vol1y,
    }
