from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd

from quant.data.loader import load_prices_long, to_wide_adj_close
from quant.strategy.universe import load_universe
from quant.strategy.policy import load_params, build_signals
from quant.data.return_provider import ReturnProvider
from quant.backtest.engine import run_backtest
from quant.report.metrics import cagr, max_drawdown, calmar, turnover, last_n_years_metrics
from quant.report.artifacts import save_result, ensure_dir
from quant.report.generate import generate_report
from quant.report.plots import equity_and_drawdown_plots

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prices", required=True)
    ap.add_argument("--config", required=True, help="Universe yml")
    ap.add_argument("--params", required=True, nargs="+", help="Exactly 3 params json files")
    ap.add_argument("--out-dir", required=True, help="Output directory root")
    return ap.parse_args()

def _read_prices(prices_path: str) -> pd.DataFrame:
    df_long = load_prices_long(prices_path)
    return to_wide_adj_close(df_long)

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

    m10 = last_n_years_metrics(equity, years=10)

    metrics = {
        "CAGR": cagr(equity),
        "MDD": max_drawdown(equity),
        "Calmar": calmar(equity),
        "Turnover": turnover(weights),
        "Start": str(pd.to_datetime(equity.index.min()).date()),
        "End": str(pd.to_datetime(equity.index.max()).date()),
        "Last10Y_Start": m10["start"],
        "Last10Y_End": m10["end"],
        "Last10Y_SeedMultiple": m10["multiple"],
        "Last10Y_CAGR": m10["cagr"],
        "Last10Y_MDD": m10["mdd"],
        "ParamsFile": params_path,
    }

    ensure_dir(out_dir)
    save_result(out_dir, equity, weights, trades, metrics)

    # Pretty report + plots
    generate_report(out_dir, equity, weights, trades)
    equity_and_drawdown_plots(equity, out_dir)

    return metrics

def main():
    args = parse_args()
    if len(args.params) != 3:
        raise SystemExit(f"--params expects exactly 3 files, got {len(args.params)}")

    out_root = Path(args.out_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    prices = _read_prices(args.prices)

    rows = []
    for p in args.params:
        stem = Path(p).stem  # folder name (no 'round' required)
        out_dir = str(out_root / stem)
        m = run_one(prices, args.config, p, out_dir)
        m["Name"] = stem
        rows.append(m)

    df = pd.DataFrame(rows)
    df = df[[
        "Name",
        "Last10Y_SeedMultiple", "Last10Y_CAGR", "Last10Y_MDD",
        "CAGR", "MDD", "Calmar", "Turnover",
        "Start", "End", "ParamsFile"
    ]].sort_values(by="Last10Y_CAGR", ascending=False)

    df.to_csv(out_root / "summary.csv", index=False)
    print(df.to_string(index=False))
    print(f"\nSaved summary -> {out_root / 'summary.csv'}")
    print(f"Saved runs under -> {out_root}")

if __name__ == "__main__":
    main()
