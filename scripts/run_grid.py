from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Any, Optional, List

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
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _deep_merge(dst: Dict[str, Any], src: Dict[str, Any]) -> Dict[str, Any]:
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
    ap.add_argument("--grid", required=True, help="Grid yml (e.g. config/grid_hysteresis_push.yml)")
    ap.add_argument("--out-dir", required=True, help="e.g. out/grid_hysteresis_push")
    ap.add_argument(
        "--portfolio-config",
        required=False,
        default=None,
        help="Optional yml to override params.selection (e.g. config/portfolio_top1.yml)",
    )
    return ap.parse_args()


def _get_nested(d: Dict[str, Any], path: str, default=None):
    cur = d
    for p in path.split("."):
        if not isinstance(cur, dict) or p not in cur:
            return default
        cur = cur[p]
    return cur


def main():
    args = parse_args()
    ensure_dir(args.out_dir)
    outp = Path(args.out_dir)

    # --- Resolve paths and dump debug info early ---
    script_path = Path(__file__).resolve()
    repo_root = script_path.parents[1]  # .../multiaset_quant_engine
    base_params_path = repo_root / "config" / "default_params.yml"

    debug: Dict[str, Any] = {
        "cwd": str(Path.cwd()),
        "script_path": str(script_path),
        "repo_root_guess": str(repo_root),
        "prices_path_arg": args.prices,
        "universe_config_arg": args.config,
        "grid_config_arg": args.grid,
        "portfolio_config_arg": args.portfolio_config,
        "base_params_path_guess": str(base_params_path),
        "base_params_exists": base_params_path.exists(),
    }

    print("[DEBUG] cwd:", debug["cwd"])
    print("[DEBUG] script_path:", debug["script_path"])
    print("[DEBUG] repo_root_guess:", debug["repo_root_guess"])
    print("[DEBUG] base_params_path_guess:", debug["base_params_path_guess"], "exists:", debug["base_params_exists"])
    print("[DEBUG] universe_config_arg:", args.config)
    print("[DEBUG] grid_config_arg:", args.grid)
    if args.portfolio_config:
        print("[DEBUG] portfolio_config_arg:", args.portfolio_config)

    # Universe
    uni = load_universe(args.config)

    # Base params
    if not base_params_path.exists():
        raise SystemExit(f"Cannot find base params file at: {base_params_path}")

    base_params = _load_yaml(str(base_params_path))
    debug["base_params_selection_has_hysteresis"] = _get_nested(base_params, "selection.hysteresis") is not None
    debug["base_params_selection_hysteresis"] = _get_nested(base_params, "selection.hysteresis", None)

    # Grid config
    grid_cfg = _load_yaml(args.grid)
    wf_cfg = grid_cfg.get("walkforward", {}) or {}
    grid = grid_cfg.get("grid", {}) or {}

    debug["grid_has_selection_hysteresis_enabled"] = "selection.hysteresis.enabled" in grid
    debug["grid_has_selection_hysteresis_min_improve"] = "selection.hysteresis.min_improve" in grid
    debug["grid_selection_hysteresis_enabled_values"] = grid.get("selection.hysteresis.enabled", None)
    debug["grid_selection_hysteresis_min_improve_values"] = grid.get("selection.hysteresis.min_improve", None)

    print("[DEBUG] base_params selection.hysteresis exists?:", debug["base_params_selection_has_hysteresis"])
    print("[DEBUG] grid has selection.hysteresis.enabled?:", debug["grid_has_selection_hysteresis_enabled"])
    print("[DEBUG] grid has selection.hysteresis.min_improve?:", debug["grid_has_selection_hysteresis_min_improve"])
    print("[DEBUG] grid selection.hysteresis.enabled values:", debug["grid_selection_hysteresis_enabled_values"])
    print("[DEBUG] grid selection.hysteresis.min_improve values:", debug["grid_selection_hysteresis_min_improve_values"])

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
    debug["param_sets_count"] = len(param_sets)

    # --- verify hysteresis landed in param_sets ---
    enabled_vals = set()
    min_improve_vals = set()
    for p in param_sets:
        hv = _get_nested(p, "selection.hysteresis.enabled", None)
        mv = _get_nested(p, "selection.hysteresis.min_improve", None)
        if hv is not None:
            enabled_vals.add(bool(hv))
        if mv is not None:
            min_improve_vals.add(float(mv))

    debug["param_sets_hysteresis_enabled_unique"] = sorted(list(enabled_vals))
    debug["param_sets_hysteresis_min_improve_unique"] = sorted(list(min_improve_vals))

    print("[DEBUG] param_sets_count:", debug["param_sets_count"])
    print("[DEBUG] param_sets hysteresis.enabled unique:", debug["param_sets_hysteresis_enabled_unique"])
    print("[DEBUG] param_sets hysteresis.min_improve unique:", debug["param_sets_hysteresis_min_improve_unique"])

    # save debug json
    (outp / "debug_paths.json").write_text(json.dumps(debug, indent=2), encoding="utf-8")

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
            params = p
            if portfolio_cfg:
                params = json.loads(json.dumps(p))  # deep copy
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

    wf_df.to_csv(outp / "wf_results.csv", index=False)
    try:
        wf_df.to_parquet(outp / "wf_results.parquet", index=False)
    except Exception:
        pass

    summ = summarize_by_param(wf_df[["param_id", "window_id", "cagr", "mdd", "calmar", "turnover"]])
    summ.to_csv(outp / "param_summary.csv", index=False)

    best_pid = str(summ.iloc[0]["param_id"])

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

    (outp / "best_params.json").write_text(json.dumps(best_params, indent=2), encoding="utf-8")

    signals = build_signals(prices_full, kind_map, uni.regime_ticker, uni.cash_proxy, best_params)
    equity, daily_weights, trades = run_backtest(prices_full, signals, rp_full, best_params["costs"])
    generate_report(str(outp), equity, daily_weights, trades)

    print(f"Best param_id: {best_pid}")
    print("Done.")


if __name__ == "__main__":
    main()