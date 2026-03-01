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


def _get(d: dict, path: list[str], default=None):
    cur = d
    for k in path:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


def params_to_row(p: dict) -> dict:
    """
    Pick key knobs so param_summary is readable.
    Also include full params_json for perfect reproducibility.
    """
    row = {
        # rebalance / selection
        "reb_freq": _get(p, ["rebalance", "frequency"]),
        "reb_when": _get(p, ["rebalance", "when"]),
        "top_n": _get(p, ["selection", "top_n"]),
        "weighting": _get(p, ["selection", "weighting"]),

        # regime / filters
        "regime_ma_days": _get(p, ["filters", "regime_ma_days"]),
        "abs_mom_min": _get(p, ["filters", "abs_mom_min"]),
        "trend_fast": _get(p, ["filters", "trend_fast"]),
        "trend_slow": _get(p, ["filters", "trend_slow"]),

        # weights (momentum blend)
        "w_1m": _get(p, ["weights", "w_1m"]),
        "w_6m": _get(p, ["weights", "w_6m"]),
        "w_12m": _get(p, ["weights", "w_12m"]),

        # leverage
        "lev_enabled": _get(p, ["leverage", "enabled"]),
        "equity_only": _get(p, ["leverage", "equity_only"]),
        "use_trend_confirm": _get(p, ["leverage", "use_trend_confirm"]),
        "use_vol_target": _get(p, ["leverage", "use_vol_target"]),
        "vol_target_annual": _get(p, ["leverage", "vol_target_annual"]),
        "vol_lookback_days": _get(p, ["leverage", "vol_lookback_days"]),
        "max_leverage": _get(p, ["leverage", "max_leverage"]),
        "gear_cut_2x": _get(p, ["leverage", "gear_cut_2x"]),
        "gear_cut_3x": _get(p, ["leverage", "gear_cut_3x"]),

        # hybrid bull 3x
        "use_hybrid_bull_3x": _get(p, ["leverage", "use_hybrid_bull_3x"]),
        "bull_vol_ticker": _get(p, ["leverage", "bull_vol_ticker"]),
        "bull_vol_lookback_days": _get(p, ["leverage", "bull_vol_lookback_days"]),
        "bull_vol_max_annual": _get(p, ["leverage", "bull_vol_max_annual"]),
        "force_3x_assets": json.dumps(_get(p, ["leverage", "force_3x_assets"]), ensure_ascii=False),

        # costs
        "buy_cost": _get(p, ["costs", "buy"]),
        "sell_cost": _get(p, ["costs", "sell"]),

        # full reproducibility
        "params_json": json.dumps(p, ensure_ascii=False),
    }
    return row


def main():
    args = parse_args()
    t0 = time.time()

    uni = load_universe(args.config)
    grid_cfg = yaml.safe_load(open(args.grid, "r", encoding="utf-8"))

    base_params = yaml.safe_load(
        open(Path(__file__).resolve().parents[1] / "config" / "default_params.yml", "r", encoding="utf-8")
    )
    param_sets = make_param_sets(base_params, grid_cfg["grid"])

    ensure_dir(args.out_dir)
    outp = Path(args.out_dir)

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

    if not len(summary):
        (outp / "param_summary.csv").write_text("", encoding="utf-8")
        print("[run_grid] no results", flush=True)
        return

    # ✅ Build param_id -> params columns dataframe
    params_rows = []
    for p in param_sets:
        pid = param_id(p)
        r = {"param_id": pid}
        r.update(params_to_row(p))
        params_rows.append(r)
    params_df = pd.DataFrame(params_rows)

    # ✅ Merge into param_summary
    summary = summary.merge(params_df, on="param_id", how="left")

    summary_path = outp / "param_summary.csv"
    summary.to_csv(summary_path, index=False)

    # pick best
    best_pid = summary.iloc[0]["param_id"]
    best_row = params_df.loc[params_df["param_id"] == best_pid].iloc[0]
    best_params = json.loads(best_row["params_json"])
    (outp / "best_params.json").write_text(json.dumps(best_params, indent=2), encoding="utf-8")

    print("[run_grid] Running full backtest for BEST params...", flush=True)

    signals_best = build_signals(prices_full, kind_map, uni.regime_ticker, uni.cash_proxy, best_params)
    equity_full, weights_full, trades_full = run_backtest(prices_full, signals_best, rp_full, best_params["costs"])

    generate_report(str(outp), equity_full, weights_full, trades_full)

    print(f"[run_grid] done -> {outp}", flush=True)


if __name__ == "__main__":
    main()