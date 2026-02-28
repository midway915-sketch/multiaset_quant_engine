from __future__ import annotations
import argparse
import json
import math
from pathlib import Path
from typing import Tuple, Dict, Any

import numpy as np
import pandas as pd


def _read_equity_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)

    # normalize column names
    cols = {c: c.strip().lower() for c in df.columns}
    df = df.rename(columns=cols)

    # try to find date-like column
    date_col = None
    for c in ["date", "dt", "timestamp", "time"]:
        if c in df.columns:
            date_col = c
            break

    if date_col is None:
        # assume first column is date
        date_col = df.columns[0]

    # parse date
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.dropna(subset=[date_col]).sort_values(date_col)

    # try to find equity column
    eq_col = None
    for c in ["equity", "nav", "value", "portfolio_value"]:
        if c in df.columns:
            eq_col = c
            break
    if eq_col is None:
        # fallback: last numeric column
        num_cols = [c for c in df.columns if c != date_col and pd.api.types.is_numeric_dtype(df[c])]
        if not num_cols:
            raise ValueError(f"Cannot find equity column in {path}. Columns={list(df.columns)}")
        eq_col = num_cols[-1]

    out = df[[date_col, eq_col]].rename(columns={date_col: "date", eq_col: "equity"}).copy()
    out = out.dropna(subset=["equity"])
    return out


def _max_drawdown(equity: pd.Series) -> float:
    # equity must be positive
    s = equity.astype(float)
    peak = s.cummax()
    dd = (s / peak) - 1.0
    return float(dd.min()) if len(dd) else float("nan")


def _cagr_from_multiple(multiple: float, years: float) -> float:
    if not np.isfinite(multiple) or multiple <= 0 or years <= 0:
        return float("nan")
    return float(multiple ** (1.0 / years) - 1.0)


def _window_metrics(eq_df: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> Dict[str, float]:
    w = eq_df[(eq_df["date"] >= start) & (eq_df["date"] <= end)].copy()
    if len(w) < 50:
        return {"multiple": float("nan"), "cagr": float("nan"), "mdd": float("nan")}

    e0 = float(w["equity"].iloc[0])
    e1 = float(w["equity"].iloc[-1])
    multiple = e1 / e0 if e0 > 0 else float("nan")

    years = (w["date"].iloc[-1] - w["date"].iloc[0]).days / 365.25
    cagr = _cagr_from_multiple(multiple, years)
    mdd = _max_drawdown(w["equity"])
    return {"multiple": float(multiple), "cagr": float(cagr), "mdd": float(mdd)}


def rolling10y_stats(eq_df: pd.DataFrame, window_years: int = 10) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    eq_df = eq_df.sort_values("date").reset_index(drop=True)

    start_min = eq_df["date"].min()
    end_max = eq_df["date"].max()

    rows = []
    # end dates: every row date (but we can thin if needed)
    for end in eq_df["date"]:
        start = end - pd.DateOffset(years=window_years)
        if start < start_min:
            continue
        m = _window_metrics(eq_df, start, end)
        if not np.isfinite(m["cagr"]):
            continue
        rows.append({
            "date": end,
            "roll10y_multiple": m["multiple"],
            "roll10y_cagr": m["cagr"],
            "roll10y_mdd": m["mdd"],
        })

    roll = pd.DataFrame(rows).sort_values("date").reset_index(drop=True)

    # summary stats
    stats = {
        "n_windows": int(len(roll)),
        "median_cagr": float(roll["roll10y_cagr"].median()) if len(roll) else float("nan"),
        "median_mdd": float(roll["roll10y_mdd"].median()) if len(roll) else float("nan"),
        "median_calmar": float((roll["roll10y_cagr"] / (-roll["roll10y_mdd"])).replace([np.inf, -np.inf], np.nan).median()) if len(roll) else float("nan"),
        "worst_cagr": float(roll["roll10y_cagr"].min()) if len(roll) else float("nan"),
        "p10_cagr": float(roll["roll10y_cagr"].quantile(0.10)) if len(roll) else float("nan"),
        "p90_cagr": float(roll["roll10y_cagr"].quantile(0.90)) if len(roll) else float("nan"),
    }

    # last 10y (single trailing window)
    last_end = end_max
    last_start = last_end - pd.DateOffset(years=window_years)
    last = _window_metrics(eq_df, last_start, last_end)
    stats.update({
        "last10y_cagr": last["cagr"],
        "last10y_multiple": last["multiple"],
        "last10y_mdd": last["mdd"],
        "last10y_start": str(last_start.date()),
        "last10y_end": str(last_end.date()),
    })

    return roll, stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--equity-csv", required=True)
    ap.add_argument("--out-rolling-csv", default="rolling10y.csv")
    ap.add_argument("--out-stats-json", default="rolling10y_stats.json")
    args = ap.parse_args()

    eq = _read_equity_csv(args.equity_csv)
    roll_df, stats = rolling10y_stats(eq, window_years=10)

    Path(args.out_rolling_csv).parent.mkdir(parents=True, exist_ok=True)
    roll_df.to_csv(args.out_rolling_csv, index=False)

    Path(args.out_stats_json).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_stats_json, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    print("equity last:", eq["date"].max())
    print("rolling windows:", stats.get("n_windows"))


if __name__ == "__main__":
    main()