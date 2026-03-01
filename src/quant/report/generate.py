from __future__ import annotations
from pathlib import Path
import json
import pandas as pd

from quant.report.metrics import cagr, max_drawdown, calmar, turnover, yearly_returns, last_n_years_metrics

def generate_report(out_dir: str, equity: pd.Series, weights: pd.DataFrame, trades: pd.DataFrame):
    outp = Path(out_dir)
    outp.mkdir(parents=True, exist_ok=True)

    # Metrics (full period)
    metrics = {
        "CAGR": cagr(equity),
        "MDD": max_drawdown(equity),
        "Calmar": calmar(equity),
        "AvgDailyTurnover": turnover(weights),
        "Start": str(equity.index.min().date()),
        "End": str(equity.index.max().date()),
        "NumDays": int(len(equity)),
        "NumRebalances": int(len(trades)) if trades is not None else 0,
    }

    # Last 10Y window metrics (requested summary fields)
    m10 = last_n_years_metrics(equity, years=10)
    metrics.update({
        "Last10Y_Start": m10["start"],
        "Last10Y_End": m10["end"],
        "Last10Y_SeedMultiple": m10["multiple"],
        "Last10Y_CAGR": m10["cagr"],
        "Last10Y_MDD": m10["mdd"],
    })

    (outp / "metrics_pretty.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    # Yearly returns table
    yr = yearly_returns(equity).rename("year_return").to_frame()
    yr.to_csv(outp / "yearly_returns.csv", index=True)

    # Rebalance log (flatten key fields)
    if trades is not None and len(trades):
        t = trades.copy()
        # weights/gears columns may be dicts; keep as JSON strings
        for col in ["weights", "gears", "assets"]:
            if col in t.columns:
                t[col] = t[col].apply(lambda x: json.dumps(x, ensure_ascii=False) if isinstance(x, (dict, list)) else x)
        t.to_csv(outp / "rebalance_log.csv", index=False)