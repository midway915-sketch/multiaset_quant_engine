from __future__ import annotations

import argparse
import json
from pathlib import Path
import yaml
import pandas as pd

from quant.data.loader import load_prices_long, to_wide_adj_close
from quant.strategy.universe import load_universe
from quant.strategy.policy import build_signals
from quant.data.return_provider import ReturnProvider
from quant.backtest.engine import run_backtest
from quant.grid.grid import make_param_sets, param_id
from quant.report.artifacts import ensure_dir
from quant.report.generate import generate_report


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prices", required=True, help="e.g. data/prices.csv")
    ap.add_argument("--config", required=True, help="Universe yml (e.g. config/universe_round10.yml)")
    ap.add_argument("--grid", required=True, help="Grid yml used to generate param_sets (e.g. config/grid_cagr25_fast.yml)")
    ap.add_argument("--param-id", required=True, help="param_id to rerun (e.g. a74bf54c38)")
    ap.add_argument("--out-dir", required=True, help="e.g. out/rerun_a74bf54c38")
    return ap.parse_args()


def main():
    args = parse_args()

    uni = load_universe(args.config)
    grid_cfg = yaml.safe_load(open(args.grid, "r", encoding="utf-8"))

    base_params_path = Path(__file__).resolve().parents[1] / "config" / "default_params.yml"
    base_params = yaml.safe_load(open(base_params_path, "r", encoding="utf-8"))

    # rebuild all param sets from the same grid, then pick the one matching param_id
    param_sets = make_param_sets(base_params, grid_cfg["grid"])
    target = None
    for p in param_sets:
        if param_id(p) == args.param_id:
            target = p
            break
    if target is None:
        raise SystemExit(f"param_id not found in this grid: {args.param_id}")

    # load prices
    df_long = load_prices_long(args.prices)
    prices_full = to_wide_adj_close(df_long)

    needed = [t for t in uni.tickers if t in prices_full.columns]
    prices_full = prices_full[needed].dropna(how="all")

    kind_map = uni.kind_map()
    rp_full = ReturnProvider(
        prices_wide=prices_full,
        inception=uni.inception,
        annual_fees=uni.annual_fees,
        leverage_maps=uni.leverage_maps(),
    )

    ensure_dir(args.out_dir)
    outp = Path(args.out_dir)

    # save the exact params we reran
    (outp / "rerun_param_id.txt").write_text(args.param_id, encoding="utf-8")
    (outp / "best_params.json").write_text(json.dumps(target, indent=2), encoding="utf-8")

    # full backtest + report (creates picks_top2_weekly.csv)
    signals = build_signals(prices_full, kind_map, uni.regime_ticker, uni.cash_proxy, target)
    equity, weights, trades = run_backtest(prices_full, signals, rp_full, target["costs"])
    generate_report(str(outp), equity, weights, trades)

    print(f"[rerun_param_id] done -> {outp}")


if __name__ == "__main__":
    main()