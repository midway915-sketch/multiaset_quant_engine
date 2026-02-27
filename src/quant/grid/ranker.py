from __future__ import annotations
import pandas as pd
import numpy as np

def summarize_by_param(results: pd.DataFrame) -> pd.DataFrame:
    # results rows: [param_id, window_id, cagr, mdd, calmar, turnover]
    grp = results.groupby("param_id")
    out = pd.DataFrame({
        "median_cagr": grp["cagr"].median(),
        "median_mdd": grp["mdd"].median(),
        "median_calmar": grp["calmar"].median(),
        "worst_cagr": grp["cagr"].min(),
        "mean_turnover": grp["turnover"].mean(),
        "n_windows": grp.size(),
    }).reset_index()
    return out.sort_values("median_calmar", ascending=False)
