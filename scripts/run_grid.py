from __future__ import annotations
import argparse
from pathlib import Path
import yaml
import pandas as pd
import json
import time

from quant.data.loader import load_prices_long, to_wide_adj_close
from quant.strategy.universe import load_universe
from quant.strategy.policy import build_signals
from quant.data.return_provider import ReturnProvider
from quant.backtest.engine import run_backtest
from quant.report.metrics import cagr, max_drawdown, calmar, turnover
from quant.grid.grid import make_param_sets, param_id
from quant.grid.walkforward import make_walkforward_windows
from quant.grid.ranker import summarize_by_param
from quant.report.artifacts import ensure_dir
from quant.report.generate import generate_report


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prices", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--grid", required=True)
    ap.add_argument("--out-dir", required=True)
    return ap.parse_args()


def slice_prices(prices: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    return prices.loc[(prices.index >= start) & (prices.index <= end)]


def save_df(df: pd.DataFrame, out: Path, stem: str):
    pq = out / f"{stem}.parquet"
    try:
        df.to_parquet(pq, index=False)
    except Exception:
        df.to_csv(out / f"{stem}.csv", index=False)


def _safe_load_json(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def main():
    args = parse_args()
    t0 = time.time()

    uni = load_universe(args.config)
    grid_cfg = yaml.safe_load(open(args.grid, "r", encoding="utf-8"))

    base_params = yaml.safe_load(
        open(Path(__file__).resolve().parents[1] / "config" / "default_params.yml", "r", encoding="utf-8")
    )
    param_sets = make_param_sets(base_params, grid_cfg["grid"])

    df_long = load_prices_long(args.prices)
    prices_full = to_wide_adj_close(df_long)

    needed = [t for t in uni.tickers if t in prices_full.columns]
    prices_full = prices_full[needed].dropna(how="all")

    windows = make_walkforward_windows(
        prices_full.index,
        train_years=grid_cfg["walkforward"]["train_years"],
        test_years=grid_cfg["walkforward"]["test_years"],
        step_years=grid_cfg["walkforward"]["step_years"],
    )

    ensure_dir(args.out_dir)
    outp = Path(args.out_dir)

    kind_map = uni.kind_map()
    rp_full = ReturnProvider(
        prices_wide=prices_full,
        inception=uni.inception,
        annual_fees=uni.annual_fees,
        leverage_maps=uni.leverage_maps(),
    )

    total_params = len(param_sets)
    total_windows = len(windows)
    total_eval = total_params * total_windows

    print(f"[run_grid] params={total_params}, windows={total_windows}, evals~={total_eval}", flush=True)

    results_rows = []
    done = 0

    for pi, p in enumerate(param_sets, start=1):
        pid = param_id(p)
        signals_all = build_signals(prices_full, kind_map, uni.regime_ticker, uni.cash_proxy, p)

        print(f"[run_grid] ({pi}/{total_params}) param_id={pid}", flush=True)

        for wi, w in enumerate(windows):
            ptest = slice_prices(prices_full, w.test_start, w.test_end)
            if len(ptest) < 300:
                done += 1
                continue

            signals = signals_all[
                (pd.to_datetime(signals_all["apply_date"]) >= w.test_start) &
                (pd.to_datetime(signals_all["apply_date"]) <= w.test_end)
            ]

            equity, weights, trades = run_backtest(ptest, signals, rp_full, p["costs"])

            results_rows.append({
                "param_id": pid,
                "window_id": wi,
                "test_start": str(w.test_start.date()),
                "test_end": str(w.test_end.date()),
                "cagr": cagr(equity),
                "mdd": max_drawdown(equity),
                "calmar": calmar(equity),
                "turnover": turnover(weights),
            })

            done += 1
            if done % 50 == 0:
                elapsed = time.time() - t0
                rate = done / elapsed if elapsed > 0 else 0
                eta = (total_eval - done) / rate if rate > 0 else 0
                print(f"[run_grid] {done}/{total_eval} ({done/total_eval:.1%}) ETA {eta/60:.1f}m", flush=True)

    results = pd.DataFrame(results_rows)
    save_df(results, outp, "wf_results")

    summary = summarize_by_param(results) if len(results) else pd.DataFrame()
    summary_path = outp / "param_summary.csv"
    summary.to_csv(summary_path, index=False)

    if not len(summary):
        print("[run_grid] no results")
        return

    best_pid = summary.iloc[0]["param_id"]
    best = None
    for p in param_sets:
        if param_id(p) == best_pid:
            best = p
            break

    (outp / "best_params.json").write_text(json.dumps(best, indent=2), encoding="utf-8")

    print("[run_grid] Running full backtest for BEST params...", flush=True)

    # BEST PARAM FULL BACKTEST (전체 구간)
    signals_best = build_signals(prices_full, kind_map, uni.regime_ticker, uni.cash_proxy, best)
    equity_full, weights_full, trades_full = run_backtest(prices_full, signals_best, rp_full, best["costs"])

    # 여기서 metrics_pretty.json + picks_top2_weekly.csv 생성됨
    generate_report(str(outp), equity_full, weights_full, trades_full)

    # ✅ metrics_pretty.json에서 Last10Y를 읽어서 param_summary의 best 행에 주입
    metrics = _safe_load_json(outp / "metrics_pretty.json")
    last10_cols = {
        "Last10Y_SeedMultiple": metrics.get("Last10Y_SeedMultiple"),
        "Last10Y_CAGR": metrics.get("Last10Y_CAGR"),
        "Last10Y_MDD": metrics.get("Last10Y_MDD"),
    }

    try:
        ps = pd.read_csv(summary_path)
        for c in ["Last10Y_SeedMultiple", "Last10Y_CAGR", "Last10Y_MDD"]:
            if c not in ps.columns:
                ps[c] = pd.NA
        ps.loc[ps["param_id"] == best_pid, "Last10Y_SeedMultiple"] = last10_cols["Last10Y_SeedMultiple"]
        ps.loc[ps["param_id"] == best_pid, "Last10Y_CAGR"] = last10_cols["Last10Y_CAGR"]
        ps.loc[ps["param_id"] == best_pid, "Last10Y_MDD"] = last10_cols["Last10Y_MDD"]
        ps.to_csv(summary_path, index=False)
        print("[run_grid] injected Last10Y_* into param_summary (best row only)", flush=True)
    except Exception as e:
        print(f"[run_grid] failed to inject Last10Y into param_summary: {e}", flush=True)

    print(f"[run_grid] done -> {outp}", flush=True)


if __name__ == "__main__":
    main()