from __future__ import annotations

import argparse
from pathlib import Path
import re
import json
import pandas as pd

# NOTE:
# - This script assumes your project already has the same core functions used in your other runners.
# - If your internal module paths differ, keep the logic and adjust imports to match your repo.

from quant.data.loader import load_prices_long, to_wide_adj_close
from quant.strategy.universe import load_universe
from quant.strategy.policy import load_params, build_signals
from quant.data.return_provider import ReturnProvider
from quant.backtest.engine import run_backtest
from quant.report.metrics import cagr, max_drawdown, calmar, turnover, last_n_years_metrics
from quant.report.artifacts import save_result, ensure_dir
from quant.report.generate import generate_report

try:
    # optional: if you have plots
    from quant.report.plots import equity_and_drawdown_plots
except Exception:
    equity_and_drawdown_plots = None


def _read_prices(prices_path: str) -> pd.DataFrame:
    df_long = load_prices_long(prices_path)
    return to_wide_adj_close(df_long)


def _resolve_params_path(p: str, base_dir: str) -> str:
    """
    Allow passing:
      - full path: configs/best/round5.json
      - filename only: round5.json
    and resolve under base_dir if not already a path.
    """
    pp = Path(p)
    if pp.exists():
        return str(pp)

    # If user provided only name, resolve inside base_dir
    base = Path(base_dir)
    cand = base / p
    if cand.exists():
        return str(cand)

    # Try adding .json if omitted
    if pp.suffix == "":
        cand2 = base / f"{p}.json"
        if cand2.exists():
            return str(cand2)

    raise FileNotFoundError(f"Params file not found: {p} (searched: {cand})")


def _run_name_from_params_path(params_path: str) -> str:
    """
    Folder/run name should NOT contain 'round' and should not collide with your existing runs.
    Example:
      configs/best/round7b.json -> 'best_7b'
      configs/best/round5.json  -> 'best_5'
    """
    stem = Path(params_path).stem  # e.g. "round7b"
    m = re.match(r"round(.+)", stem, re.IGNORECASE)
    suffix = m.group(1) if m else stem
    suffix = re.sub(r"[^a-zA-Z0-9_-]+", "_", suffix).strip("_")
    return f"best_{suffix}"


def run_one(prices: pd.DataFrame, uni_config: str, params_path: str, out_dir: str) -> dict:
    uni = load_universe(uni_config)
    params = load_params(params_path)

    needed = [t for t in uni.tickers if t in prices.columns]
    px = prices[needed].dropna(how="all")

    kind_map = uni.kind_map()
    signals = build_signals(px, kind_map, uni.regime_ticker, uni.cash_proxy, params)

    rp = ReturnProvider(
        prices_wide=px,
        inception=uni.inception,
        annual_fees=uni.annual_fees,
        leverage_maps=uni.leverage_maps(),
    )

    equity, weights, trades = run_backtest(px, signals, rp, params["costs"])

    # ✅ exact last-10Y slice from equity curve (not an estimate)
    m10 = last_n_years_metrics(equity, years=10)

    metrics = {
        "ParamsFile": params_path,

        # Full period
        "CAGR": cagr(equity),
        "MDD": max_drawdown(equity),
        "Calmar": calmar(equity),
        "Turnover": turnover(weights),

        # Last 10Y (exact slice)
        "Last10Y_Start": m10["start"],
        "Last10Y_End": m10["end"],
        "Last10Y_SeedMultiple": m10["multiple"],
        "Last10Y_CAGR": m10["cagr"],
        "Last10Y_MDD": m10["mdd"],
    }

    ensure_dir(out_dir)
    save_result(out_dir, equity, weights, trades, metrics)
    generate_report(out_dir, equity, weights, trades)

    if equity_and_drawdown_plots is not None:
        try:
            equity_and_drawdown_plots(equity, out_dir)
        except Exception:
            pass

    return metrics


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prices", required=True, help="prices file (csv/parquet) as your existing runners expect")
    ap.add_argument("--config", required=True, help="Universe yml (ex: config/universe_round10.yml)")
    ap.add_argument("--params-base-dir", default="configs/best", help="where round*.json lives")
    ap.add_argument(
        "--params",
        required=True,
        nargs="+",
        help="Exactly 3 params JSONs. You can pass full path or filename (e.g. round5.json).",
    )
    ap.add_argument("--out-dir", required=True, help="Output directory root (will create subfolders)")
    return ap.parse_args()


def main():
    args = parse_args()
    if len(args.params) != 3:
        raise SystemExit(f"--params expects exactly 3 files, got {len(args.params)}")

    out_root = Path(args.out_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    prices = _read_prices(args.prices)

    rows = []
    for p in args.params:
        resolved = _resolve_params_path(p, args.params_base_dir)
        run_name = _run_name_from_params_path(resolved)
        out_dir = str(out_root / run_name)

        m = run_one(prices, args.config, resolved, out_dir)
        m["Name"] = run_name
        rows.append(m)

    df = pd.DataFrame(rows)
    cols = [
        "Name",
        "Last10Y_SeedMultiple", "Last10Y_CAGR", "Last10Y_MDD",
        "CAGR", "MDD", "Calmar", "Turnover",
        "ParamsFile",
    ]
    df = df[cols].sort_values(by="Last10Y_CAGR", ascending=False)

    summary_path = out_root / "summary.csv"
    df.to_csv(summary_path, index=False)

    print(df.to_string(index=False))
    print(f"\nSaved summary -> {summary_path}")
    print(f"Saved runs under -> {out_root}")


if __name__ == "__main__":
    main()
