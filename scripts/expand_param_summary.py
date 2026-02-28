from __future__ import annotations
import argparse
import glob
import math
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd


def read_equity_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    cols = {c: c.strip().lower() for c in df.columns}
    df = df.rename(columns=cols)

    date_col = None
    for c in ["date", "dt", "timestamp", "time"]:
        if c in df.columns:
            date_col = c
            break
    if date_col is None:
        date_col = df.columns[0]

    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.dropna(subset=[date_col]).sort_values(date_col)

    eq_col = None
    for c in ["equity", "nav", "value", "portfolio_value"]:
        if c in df.columns:
            eq_col = c
            break
    if eq_col is None:
        num_cols = [c for c in df.columns if c != date_col and pd.api.types.is_numeric_dtype(df[c])]
        if not num_cols:
            raise ValueError(f"Cannot find equity column in {path}. Columns={list(df.columns)}")
        eq_col = num_cols[-1]

    out = df[[date_col, eq_col]].rename(columns={date_col: "date", eq_col: "equity"}).copy()
    out = out.dropna(subset=["equity"])
    return out


def max_drawdown(equity: pd.Series) -> float:
    s = equity.astype(float)
    peak = s.cummax()
    dd = (s / peak) - 1.0
    return float(dd.min()) if len(dd) else float("nan")


def cagr_from_multiple(multiple: float, years: float) -> float:
    if not np.isfinite(multiple) or multiple <= 0 or years <= 0:
        return float("nan")
    return float(multiple ** (1.0 / years) - 1.0)


def window_metrics(eq: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> Tuple[float, float, float]:
    w = eq[(eq["date"] >= start) & (eq["date"] <= end)].copy()
    if len(w) < 50:
        return float("nan"), float("nan"), float("nan")
    e0 = float(w["equity"].iloc[0])
    e1 = float(w["equity"].iloc[-1])
    multiple = e1 / e0 if e0 > 0 else float("nan")
    years = (w["date"].iloc[-1] - w["date"].iloc[0]).days / 365.25
    cagr = cagr_from_multiple(multiple, years)
    mdd = max_drawdown(w["equity"])
    return float(cagr), float(multiple), float(mdd)


def load_equity_for_param(param_id: str, equity_glob: str) -> Optional[str]:
    # equity_glob examples:
    #   "_out/*/{param_id}/equity.csv"
    #   "_out/{param_id}/equity.csv"
    pattern = equity_glob.format(param_id=param_id)
    matches = sorted(glob.glob(pattern))
    return matches[0] if matches else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--param-summary", required=True)
    ap.add_argument("--equity-glob", required=True,
                    help='glob pattern containing "{param_id}", e.g. "_out/**/{param_id}/equity.csv"')
    ap.add_argument("--out-csv", default="param_summary_expanded.csv")
    ap.add_argument("--window-years", type=int, default=10)
    args = ap.parse_args()

    base = pd.read_csv(args.param_summary)
    if "param_id" not in base.columns:
        raise ValueError("param_summary must include param_id column")

    rows = []
    for _, r in base.iterrows():
        pid = str(r["param_id"])
        eq_path = load_equity_for_param(pid, args.equity_glob)
        out = dict(r)

        if not eq_path:
            out.update({
                "full_cagr": np.nan, "full_multiple": np.nan, "full_mdd": np.nan,
                "last10y_cagr": np.nan, "last10y_multiple": np.nan, "last10y_mdd": np.nan,
                "equity_path": "",
            })
            rows.append(out)
            continue

        eq = read_equity_csv(eq_path)
        # full period
        full_cagr, full_mult, full_mdd = window_metrics(eq, eq["date"].min(), eq["date"].max())

        # last N years (trailing)
        end = eq["date"].max()
        start = end - pd.DateOffset(years=args.window_years)
        last_cagr, last_mult, last_mdd = window_metrics(eq, start, end)

        out.update({
            "full_cagr": full_cagr,
            "full_multiple": full_mult,
            "full_mdd": full_mdd,
            "last10y_cagr": last_cagr,
            "last10y_multiple": last_mult,
            "last10y_mdd": last_mdd,
            "equity_path": eq_path,
        })
        rows.append(out)

    outdf = pd.DataFrame(rows)
    Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
    outdf.to_csv(args.out_csv, index=False)
    print("wrote:", args.out_csv)


if __name__ == "__main__":
    main()