from __future__ import annotations

import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd


def read_equity(path: Path) -> pd.Series:
    df = pd.read_csv(path)
    if "date" not in df.columns or "equity" not in df.columns:
        raise ValueError(f"equity.csv must have columns: date,equity. got={list(df.columns)}")

    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date")
    s = pd.Series(df["equity"].astype(float).values, index=pd.DatetimeIndex(df["date"]))
    s = s[~s.index.duplicated(keep="last")]
    s = s.dropna()
    s = s[s > 0]
    return s


def compute_mdd(equity: pd.Series) -> float:
    peak = equity.cummax()
    dd = equity / peak - 1.0
    return float(dd.min())


def rolling10y_stats(equity: pd.Series, window_days: int = 3652):
    """
    True rolling 10y seed multiple & CAGR from equity curve.
    We align equity(t) with equity(t - 10y) using merge_asof.
    """
    idx = equity.index
    cur = pd.DataFrame({"date": idx, "equity": equity.values}).sort_values("date")
    past = pd.DataFrame({"date": idx, "equity_past": equity.values}).sort_values("date")

    cur["date_past_target"] = cur["date"] - pd.Timedelta(days=window_days)

    # align to nearest earlier past date (<= target)
    aligned = pd.merge_asof(
        cur.sort_values("date_past_target"),
        past,
        left_on="date_past_target",
        right_on="date",
        direction="backward",
        allow_exact_matches=True,
    )

    # ---- FIX: normalize date column after merge_asof ----
    # Depending on pandas behavior, we might get date_x/date_y or just date.
    if "date_x" in aligned.columns:
        aligned["date"] = aligned["date_x"]
    elif "date" not in aligned.columns and "date_past_target" in aligned.columns:
        # fallback: use the current date from cur (should exist as date_x in most cases)
        raise KeyError(f"merge_asof output missing 'date'/'date_x'. cols={list(aligned.columns)}")

    # compute rolling multiple & CAGR
    aligned["roll10y_multiple"] = aligned["equity"] / aligned["equity_past"]
    years = window_days / 365.25
    aligned["roll10y_cagr"] = aligned["roll10y_multiple"] ** (1.0 / years) - 1.0

    aligned = aligned.replace([np.inf, -np.inf], np.nan).dropna(subset=["roll10y_multiple", "roll10y_cagr"])
    aligned = aligned[aligned["roll10y_multiple"] > 0].copy()

    aligned = aligned.sort_values("date")[["date", "roll10y_multiple", "roll10y_cagr"]]

    stats = {}
    if len(aligned):
        stats = {
            "final_multiple_full_period": float(equity.iloc[-1] / equity.iloc[0]),
            "mdd_full_period": float(compute_mdd(equity)),
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
    ap.add_argument("--root", required=True, help="out_compare")
    ap.add_argument("--rounds", default="round5,round6,round7a,round7b,round8,round9")
    args = ap.parse_args()

    root = Path(args.root)
    rounds = [r.strip() for r in args.rounds.split(",") if r.strip()]

    rows = []
    for r in rounds:
        eq_path = root / r / "equity.csv"
        if not eq_path.exists():
            print(f"[skip] missing {eq_path}")
            continue

        eq = read_equity(eq_path)
        roll_df, stats = rolling10y_stats(eq)

        if len(roll_df):
            roll_df.to_csv(root / r / "rolling10y.csv", index=False)

        (root / r / "rolling10y_stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")

        rows.append({"round": r, **stats})

    out = pd.DataFrame(rows)
    if len(out) == 0:
        raise RuntimeError("No rounds produced stats. Check equity.csv generation step.")

    out = out.sort_values(["roll10y_cagr_median", "final_multiple_full_period"], ascending=False)

    out_csv = root / "compare_table.csv"
    out.to_csv(out_csv, index=False)
    print(f"saved -> {out_csv}")

    show = [
        "round",
        "final_multiple_full_period",
        "mdd_full_period",
        "roll10y_cagr_median",
        "roll10y_cagr_p10",
        "roll10y_cagr_min",
        "roll10y_cagr_p90",
    ]
    show = [c for c in show if c in out.columns]
    with pd.option_context("display.width", 160, "display.max_columns", 200):
        print(out[show].to_string(index=False))


if __name__ == "__main__":
    main()