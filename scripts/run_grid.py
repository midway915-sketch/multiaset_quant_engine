from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Any, Optional

import pandas as pd
import yaml

from quant.data.loader import load_prices_long, to_wide_adj_close
from quant.strategy.universe import load_universe
from quant.strategy.policy import build_signals
from quant.data.return_provider import ReturnProvider
from quant.backtest.engine import run_backtest
from quant.grid.grid import make_param_sets, param_id
from quant.grid.walkforward import make_walkforward_windows
from quant.grid.ranker import summarize_by_param
from quant.report.metrics import cagr, max_drawdown, calmar, turnover
from quant.report.generate import generate_report
from quant.report.artifacts import ensure_dir


def _load_yaml(path: str) -> Dict[str, Any]:
    return yaml.safe_load(open(path, "r", encoding="utf-8"))


def _deep_merge(dst: Dict[str, Any], src: Dict[str, Any]) -> Dict[str, Any]:
    """Deep-merge src into dst (in-place), returning dst."""
    for k, v in (src or {}).items():
        if isinstance(v, dict) and isinstance(dst.get(k), dict):
            _deep_merge(dst[k], v)
        else:
            dst[k] = v
    return dst


def _read_prices_wide(prices_path: str) -> pd.DataFrame:
    df_long = load_prices_long(prices_path)
    prices = to_wide_adj_close(df_long)
    prices.index = pd.to_datetime(prices.index)
    prices = prices.sort_index()
    return prices


def _slice_and_rebase_equity(equity: pd.Series, start: pd.Timestamp, end: pd.Timestamp) -> pd.Series:
    eq = equity.loc[(equity.index >= start) & (equity.index <= end)]
    if len(eq) < 2:
        return eq
    base = float(eq.iloc[0])
    if base == 0 or not pd.notna(base):
        return eq
    return eq / base


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prices", required=True, help="e.g. data/prices.csv")
    ap.add_argument("--config", required=True, help="Universe yml (e.g. config/universe_round10.yml)")
    ap.add_argument("--grid", required=True, help="Grid yml (e.g. config/grid_cagr_push2.yml)")
    ap.add_argument("--out-dir", required=True, help="e.g. out/grid_cagr_push2")
    ap.add_argument(
        "--portfolio-config",
        required=False,
        default=None,
        help="Optional yml to override params.selection (e.g. config/portfolio_top1.yml)",
    )
    return ap.parse_args()


def main():
    args = parse_args()
    ensure_dir(args.out_dir)
    outp = Path(args.out_dir)

    # Universe
    uni = load_universe(args.config)

    # Base params
    base_params_path = Path(__file__).resolve().parents[1] / "config" / "default_params.yml"
    base_params = _load_yaml(str(base_params_path))

    # Grid config (contains walkforward + grid)
    grid_cfg = _load_yaml(args.grid)
    wf_cfg = grid_cfg.get("walkforward", {}) or {}
    grid = grid_cfg.get("grid", {}) or {}

    # Optional portfolio override (only selection.*)
    portfolio_cfg: Optional[Dict[str, Any]] = None
    if args.portfolio_config:
        portfolio_cfg = _load_yaml(args.portfolio_config)

    # Prices
    prices_full = _read_prices_wide(args.prices)
    needed = [t for t in uni.tickers if t in prices_full.columns]
    prices_full = prices_full[needed].dropna(how="all")

    kind_map = uni.kind_map()
    rp_full = ReturnProvider(
        prices_wide=prices_full,
        inception=uni.inception,
        annual_fees=uni.annual_fees,
        leverage_maps=uni.leverage_maps(),
    )

    # Walkforward windows
    train_years = int(wf_cfg.get("train_years", 5))
    test_years = int(wf_cfg.get("test_years", 3))
    step_years = int(wf_cfg.get("step_years", 2))

    windows = make_walkforward_windows(prices_full.index, train_years, test_years, step_years)

    # Build param sets
    param_sets = make_param_sets(base_params, grid)

    wf_rows = []
    for w_i, w in enumerate(windows):
        prices_window = prices_full.loc[(prices_full.index >= w.train_start) & (prices_full.index <= w.test_end)]
        if len(prices_window) < 50:
            continue

        rp_window = ReturnProvider(
            prices_wide=prices_window,
            inception=uni.inception,
            annual_fees=uni.annual_fees,
            leverage_maps=uni.leverage_maps(),
        )

        for p in param_sets:
            # apply optional selection override
            params = p
            if portfolio_cfg:
                params = json.loads(json.dumps(p))  # deep copy via json
                _deep_merge(params, portfolio_cfg)

            pid = param_id(params)

            signals = build_signals(prices_window, kind_map, uni.regime_ticker, uni.cash_proxy, params)
            equity, daily_weights, trades = run_backtest(prices_window, signals, rp_window, params["costs"])

            eq_test = _slice_and_rebase_equity(equity, w.test_start, w.test_end)
            if len(eq_test) < 2:
                continue

            w_test = daily_weights.loc[(daily_weights.index >= w.test_start) & (daily_weights.index <= w.test_end)]
            to = turnover(w_test) if len(w_test) >= 2 else 0.0

            wf_rows.append(
                {
                    "param_id": pid,
                    "window_id": f"{w_i:02d}",
                    "train_start": str(w.train_start.date()),
                    "train_end": str(w.train_end.date()),
                    "test_start": str(w.test_start.date()),
                    "test_end": str(w.test_end.date()),
                    "cagr": cagr(eq_test),
                    "mdd": max_drawdown(eq_test),
                    "calmar": calmar(eq_test),
                    "turnover": to,
                }
            )

    wf_df = pd.DataFrame(wf_rows)
    if len(wf_df) == 0:
        raise SystemExit("No walkforward results produced. Check data coverage / dates / universe.")

    # Save wf_results
    wf_csv = outp / "wf_results.csv"
    wf_pq = outp / "wf_results.parquet"
    wf_df.to_csv(wf_csv, index=False)
    try:
        wf_df.to_parquet(wf_pq, index=False)
    except Exception:
        pass

    # Param summary
    summ = summarize_by_param(wf_df[["param_id", "window_id", "cagr", "mdd", "calmar", "turnover"]])
    summ.to_csv(outp / "param_summary.csv", index=False)

    # Pick best (top row)
    best_pid = str(summ.iloc[0]["param_id"])

    # Rebuild best params
    best_params = None
    for p in param_sets:
        params = p
        if portfolio_cfg:
            params = json.loads(json.dumps(p))
            _deep_merge(params, portfolio_cfg)
        if param_id(params) == best_pid:
            best_params = params
            break
    if best_params is None:
        raise SystemExit(f"Best param_id not found after rebuild: {best_pid}")

    # Save best params
    (outp / "best_params.json").write_text(json.dumps(best_params, indent=2), encoding="utf-8")

    # Full backtest with best params + report
    signals = build_signals(prices_full, kind_map, uni.regime_ticker, uni.cash_proxy, best_params)
    equity, daily_weights, trades = run_backtest(prices_full, signals, rp_full, best_params["costs"])
    generate_report(str(outp), equity, daily_weights, trades)

    print(f"Best param_id: {best_pid}")
    print("Done.")


if __name__ == "__main__":
    main()