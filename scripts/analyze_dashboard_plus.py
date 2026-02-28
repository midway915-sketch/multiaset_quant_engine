from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd


def _read_results(out_dir: Path) -> Tuple[pd.DataFrame, Optional[pd.DataFrame], Optional[dict]]:
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
    if not np.isfinite(cagr) or years <= 0:
        return np.nan
    if cagr <= -0.999999:
        return 0.0
    return float((1.0 + cagr) ** years)


def _recovery_years_approx(mdd: float, cagr: float) -> float:
    if not np.isfinite(mdd) or not np.isfinite(cagr):
        return np.nan
    if mdd >= 0:
        return 0.0
    if cagr <= 0:
        return np.nan
    need = 1.0 / (1.0 + mdd)
    if need <= 1.0:
        return 0.0
    return float(np.log(need) / np.log(1.0 + cagr))


def _dashboard_for_param(wf_param: pd.DataFrame) -> dict:
    years = wf_param.apply(lambda r: _years_between(r["test_start"], r["test_end"]), axis=1)
    seed_mult = [_seed_multiple(c, y) for c, y in zip(wf_param["cagr"].values, years.values)]
    rec_years = [_recovery_years_approx(m, c) for m, c in zip(wf_param["mdd"].values, wf_param["cagr"].values)]

    cagr = wf_param["cagr"].astype(float)
    mdd = wf_param["mdd"].astype(float)
    calmar = wf_param["calmar"].astype(float)
    turnover = wf_param["turnover"].astype(float)

    return {
        "n_windows": int(wf_param["window_id"].nunique()) if "window_id" in wf_param.columns else int(len(wf_param)),
        "median_cagr": float(cagr.median()),
        "p25_cagr": float(cagr.quantile(0.25)),
        "p75_cagr": float(cagr.quantile(0.75)),
        "median_mdd": float(mdd.median()),
        "median_calmar": float(calmar.median()),
        "mean_turnover": float(turnover.mean()),
        "worst_3y_cagr": float(cagr.min()),
        "worst_mdd": float(mdd.min()),
        "positive_ratio_cagr": float((cagr > 0).mean()),
        "median_seed_multiple_per_window": float(np.nanmedian(seed_mult)),
        "p10_seed_multiple_per_window": float(np.nanquantile(seed_mult, 0.10)),
        "p90_seed_multiple_per_window": float(np.nanquantile(seed_mult, 0.90)),
        "typical_10y_multiple_from_median_cagr": float(_seed_multiple(float(cagr.median()), 10.0)),
        "median_recovery_years_approx": float(np.nanmedian(rec_years)),
        "p90_recovery_years_approx": float(np.nanquantile(rec_years, 0.90)),
        "max_recovery_years_approx": float(np.nanmax(rec_years)),
    }


def _read_equity(best_full_dir: Path) -> Optional[pd.Series]:
    eq_csv = best_full_dir / "equity.csv"
    if not eq_csv.exists():
        return None
    df = pd.read_csv(eq_csv)
    if "date" not in df.columns or "equity" not in df.columns:
        return None
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date")
    s = pd.Series(df["equity"].values, index=pd.DatetimeIndex(df["date"]))
    s = s[~s.index.duplicated(keep="last")]
    return s


def _rolling_10y_stats(equity: pd.Series) -> Tuple[pd.DataFrame, dict]:
    """
    True rolling 10y seed multiple and rolling 10y CAGR from equity curve.
    Uses ~10 years = 3652 days window.
    """
    eq = equity.dropna().astype(float)
    eq = eq[eq > 0]
    if len(eq) < 2000:
        return pd.DataFrame(), {}

    # Use 10y in trading calendar sense: 3652 days (~10*365.25)
    window_days = 3652
    idx = eq.index

    # For each date, compare to date - 10y (calendar) by nearest earlier date
    # We'll use merge_asof to align past values.
    cur = pd.DataFrame({"date": idx, "equity": eq.values})
    past = pd.DataFrame({"date": idx, "equity_past": eq.values})

    cur = cur.sort_values("date")
    past = past.sort_values("date")

    # target past date = date - 10y
    cur["date_past_target"] = cur["date"] - pd.Timedelta(days=window_days)

    # align to nearest earlier past date
    aligned = pd.merge_asof(
        cur.sort_values("date_past_target"),
        past,
        left_on="date_past_target",
        right_on="date",
        direction="backward",
        suffixes=("", "_past_idx"),
    )

    aligned = aligned.rename(columns={"date_x": "date"}).drop(columns=[c for c in aligned.columns if c.startswith("date_y")], errors="ignore")

    # compute rolling multiple and CAGR
    aligned["roll10y_multiple"] = aligned["equity"] / aligned["equity_past"]
    years = window_days / 365.25
    aligned["roll10y_cagr"] = aligned["roll10y_multiple"] ** (1.0 / years) - 1.0

    # keep valid
    aligned = aligned.replace([np.inf, -np.inf], np.nan).dropna(subset=["roll10y_multiple", "roll10y_cagr"])
    aligned = aligned[aligned["roll10y_multiple"] > 0].copy()
    aligned = aligned.sort_values("date")[["date", "roll10y_multiple", "roll10y_cagr"]]

    if len(aligned) == 0:
        return aligned, {}

    stats = {
        "final_multiple_full_period": float(eq.iloc[-1] / eq.iloc[0]),
        "roll10y_multiple_median": float(aligned["roll10y_multiple"].median()),
        "roll10y_multiple_p10": float(aligned["roll10y_multiple"].quantile(0.10)),
        "roll10y_multiple_p90": float(aligned["roll10y_multiple"].quantile(0.90)),
        "roll10y_cagr_median": float(aligned["roll10y_cagr"].median()),
        "roll10y_cagr_p10": float(aligned["roll10y_cagr"].quantile(0.10)),
        "roll10y_cagr_p90": float(aligned["roll10y_cagr"].quantile(0.90)),
        "roll10y_cagr_min": float(aligned["roll10y_cagr"].min()),
        "roll10y_cagr_max": float(aligned["roll10y_cagr"].max()),
    }
    return aligned, stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", required=True, help="e.g. out_grid_round9_weekly_hybrid")
    ap.add_argument("--best-full-dir", default="", help="e.g. out_grid.../best_full (optional)")
    ap.add_argument("--top-k", type=int, default=25)
    ap.add_argument("--rank-by", default="median_calmar", choices=["median_calmar", "median_cagr"])
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    wf, ps, bp = _read_results(out_dir)

    required_cols = {"param_id", "test_start", "test_end", "cagr", "mdd", "calmar", "turnover"}
    missing = required_cols - set(wf.columns)
    if missing:
        raise ValueError(f"wf_results missing columns: {sorted(missing)}")

    if ps is None:
        ps = wf.groupby("param_id").agg(
            median_cagr=("cagr", "median"),
            median_mdd=("mdd", "median"),
            median_calmar=("calmar", "median"),
            worst_3y_cagr=("cagr", "min"),
            worst_mdd=("mdd", "min"),
            mean_turnover=("turnover", "mean"),
            n_windows=("window_id", "nunique"),
        ).reset_index()

    rank_col = args.rank_by if args.rank_by in ps.columns else ("median_calmar" if "median_calmar" in ps.columns else ps.columns[1])
    ps2 = ps.sort_values(rank_col, ascending=False).head(args.top_k)
    candidate_ids = ps2["param_id"].astype(str).tolist()

    dash_rows = []
    for pid in candidate_ids:
        wf_param = wf[wf["param_id"].astype(str) == str(pid)].copy()
        if len(wf_param) == 0:
            continue
        d = _dashboard_for_param(wf_param)
        d["param_id"] = str(pid)
        dash_rows.append(d)

    dash = pd.DataFrame(dash_rows)
    if args.rank_by in dash.columns:
        dash = dash.sort_values(args.rank_by, ascending=False)

    # Optional: true rolling 10y from equity curve
    roll_stats = {}
    roll_df = pd.DataFrame()
    if args.best_full_dir:
        eq = _read_equity(Path(args.best_full_dir))
        if eq is not None:
            roll_df, roll_stats = _rolling_10y_stats(eq)

    # Save dashboard
    out_csv = out_dir / "dashboard_summary.csv"
    dash.to_csv(out_csv, index=False)

    # Save rolling 10y
    if len(roll_df):
        roll_df.to_csv(out_dir / "rolling10y.csv", index=False)

    # Save rolling stats
    if roll_stats:
        (out_dir / "rolling10y_stats.json").write_text(json.dumps(roll_stats, indent=2), encoding="utf-8")

    print(f"Saved: {out_csv}")
    if len(roll_df):
        print(f"Saved: {out_dir / 'rolling10y.csv'}")
    if roll_stats:
        print(f"Saved: {out_dir / 'rolling10y_stats.json'}")
        print(json.dumps(roll_stats, indent=2))

    # print quick view
    cols = [
        "param_id", "n_windows",
        "median_cagr", "median_mdd", "median_calmar",
        "worst_3y_cagr", "worst_mdd",
        "positive_ratio_cagr",
        "typical_10y_multiple_from_median_cagr",
        "median_recovery_years_approx",
        "mean_turnover",
    ]
    show_cols = [c for c in cols if c in dash.columns]
    with pd.option_context("display.max_columns", 200, "display.width", 180):
        print("\n=== Dashboard (top 15) ===")
        print(dash[show_cols].head(15).to_string(index=False))

    if bp is not None:
        print("\n=== best_params.json (as stored) ===")
        print(json.dumps(bp, indent=2))


if __name__ == "__main__":
    main()