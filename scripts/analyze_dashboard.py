from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd


def _read_results(out_dir: Path) -> Tuple[pd.DataFrame, Optional[pd.DataFrame], Optional[dict]]:
    """
    Reads:
      - wf_results.csv or wf_results.parquet (required)
      - param_summary.csv (optional)
      - best_params.json (optional)
    """
    wf_csv = out_dir / "wf_results.csv"
    wf_pq = out_dir / "wf_results.parquet"

    if wf_csv.exists():
        wf = pd.read_csv(wf_csv)
    elif wf_pq.exists():
        wf = pd.read_parquet(wf_pq)
    else:
        raise FileNotFoundError(f"Missing wf_results.csv/parquet in {out_dir}")

    ps = None
    ps_path = out_dir / "param_summary.csv"
    if ps_path.exists():
        ps = pd.read_csv(ps_path)

    bp = None
    bp_path = out_dir / "best_params.json"
    if bp_path.exists():
        bp = json.loads(bp_path.read_text(encoding="utf-8"))

    return wf, ps, bp


def _years_between(start: str, end: str) -> float:
    s = pd.to_datetime(start)
    e = pd.to_datetime(end)
    days = (e - s).days
    return max(days / 365.25, 1e-9)


def _seed_multiple(cagr: float, years: float) -> float:
    # Robust against weird values
    if not np.isfinite(cagr) or years <= 0:
        return np.nan
    if cagr <= -0.999999:
        return 0.0
    return float((1.0 + cagr) ** years)


def _recovery_years_approx(mdd: float, cagr: float) -> float:
    """
    Approximate "time under water" using:
      need gain = 1/(1+mdd)
      recovery_years ≈ log(need_gain) / log(1+cagr)

    Notes:
      - This is an approximation (assumes constant CAGR during recovery).
      - If CAGR <= 0, recovery might be infinite; return NaN.
    """
    if not np.isfinite(mdd) or not np.isfinite(cagr):
        return np.nan
    if mdd >= 0:
        return 0.0
    if cagr <= 0:
        return np.nan
    need = 1.0 / (1.0 + mdd)  # mdd is negative
    if need <= 1.0:
        return 0.0
    return float(np.log(need) / np.log(1.0 + cagr))


def _dashboard_for_param(wf_param: pd.DataFrame) -> dict:
    # Infer years per window (test spans might vary slightly)
    years = wf_param.apply(lambda r: _years_between(r["test_start"], r["test_end"]), axis=1)
    seed_mult = [_seed_multiple(c, y) for c, y in zip(wf_param["cagr"].values, years.values)]
    rec_years = [_recovery_years_approx(m, c) for m, c in zip(wf_param["mdd"].values, wf_param["cagr"].values)]

    cagr = wf_param["cagr"].astype(float)
    mdd = wf_param["mdd"].astype(float)
    calmar = wf_param["calmar"].astype(float)
    turnover = wf_param["turnover"].astype(float)

    out = {
        "n_windows": int(wf_param["window_id"].nunique()) if "window_id" in wf_param.columns else int(len(wf_param)),

        # “전형적 성과”
        "median_cagr": float(cagr.median()),
        "p25_cagr": float(cagr.quantile(0.25)),
        "p75_cagr": float(cagr.quantile(0.75)),
        "median_mdd": float(mdd.median()),
        "median_calmar": float(calmar.median()),
        "mean_turnover": float(turnover.mean()),

        # “최악 구간”
        "worst_3y_cagr": float(cagr.min()),
        "worst_mdd": float(mdd.min()),
        "worst_calmar": float(calmar.min()),

        # “확률/일관성”
        "positive_ratio_cagr": float((cagr > 0).mean()),

        # “시드배수(구간별)”
        "median_seed_multiple_per_window": float(np.nanmedian(seed_mult)),
        "p10_seed_multiple_per_window": float(np.nanquantile(seed_mult, 0.10)),
        "p90_seed_multiple_per_window": float(np.nanquantile(seed_mult, 0.90)),

        # “10년 배수(전형적 CAGR로 환산)”
        "typical_10y_multiple_from_median_cagr": float(_seed_multiple(float(cagr.median()), 10.0)),

        # “회복기간(근사)”
        "median_recovery_years_approx": float(np.nanmedian(rec_years)),
        "p90_recovery_years_approx": float(np.nanquantile(rec_years, 0.90)),
        "max_recovery_years_approx": float(np.nanmax(rec_years)),
    }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", required=True, help="e.g. out_grid_round9_weekly_hybrid")
    ap.add_argument("--top-k", type=int, default=25, help="how many params to include in dashboard ranking")
    ap.add_argument("--rank-by", default="median_calmar", choices=["median_calmar", "median_cagr"],
                    help="primary ranking field (uses param_summary if available; else computed)")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    wf, ps, bp = _read_results(out_dir)

    # Basic validation
    required_cols = {"param_id", "test_start", "test_end", "cagr", "mdd", "calmar", "turnover"}
    missing = required_cols - set(wf.columns)
    if missing:
        raise ValueError(f"wf_results missing columns: {sorted(missing)}")

    # Use param_summary if present, otherwise compute a minimal one
    if ps is None:
        grp = wf.groupby("param_id").agg(
            median_cagr=("cagr", "median"),
            median_mdd=("mdd", "median"),
            median_calmar=("calmar", "median"),
            worst_3y_cagr=("cagr", "min"),
            worst_mdd=("mdd", "min"),
            mean_turnover=("turnover", "mean"),
            n_windows=("window_id", "nunique"),
        ).reset_index()
        ps = grp

    # Rank candidates
    rank_col = args.rank_by
    ps2 = ps.copy()
    if rank_col not in ps2.columns:
        # fallback
        rank_col = "median_calmar" if "median_calmar" in ps2.columns else ps2.columns[1]

    ps2 = ps2.sort_values(rank_col, ascending=False).head(args.top_k)
    candidate_ids = ps2["param_id"].astype(str).tolist()

    # Build dashboard rows
    dash_rows = []
    for pid in candidate_ids:
        wf_param = wf[wf["param_id"].astype(str) == str(pid)].copy()
        if len(wf_param) == 0:
            continue
        d = _dashboard_for_param(wf_param)
        d["param_id"] = str(pid)
        dash_rows.append(d)

    dash = pd.DataFrame(dash_rows)

    # Re-rank for the dashboard output (stable)
    if args.rank_by in dash.columns:
        dash = dash.sort_values(args.rank_by, ascending=False)

    # Save
    out_csv = out_dir / "dashboard_summary.csv"
    dash.to_csv(out_csv, index=False)

    # Print quick view
    print("\n=== Dashboard (top rows) ===")
    cols = [
        "param_id",
        "n_windows",
        "median_cagr",
        "median_mdd",
        "median_calmar",
        "worst_3y_cagr",
        "worst_mdd",
        "positive_ratio_cagr",
        "typical_10y_multiple_from_median_cagr",
        "median_recovery_years_approx",
        "mean_turnover",
    ]
    show_cols = [c for c in cols if c in dash.columns]
    with pd.option_context("display.max_columns", 200, "display.width", 180):
        print(dash[show_cols].head(15).to_string(index=False))

    print(f"\nSaved: {out_csv}")

    # Also print best_params.json (if present)
    if bp is not None:
        print("\n=== best_params.json (as stored) ===")
        print(json.dumps(bp, indent=2))


if __name__ == "__main__":
    main()