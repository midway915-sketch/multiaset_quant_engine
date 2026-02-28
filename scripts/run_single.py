from __future__ import annotations
import argparse
from pathlib import Path
import yaml
import pandas as pd

from quant.data.loader import load_prices_long, to_wide_adj_close
from quant.strategy.universe import load_universe
from quant.strategy.policy import load_params, build_signals
from quant.data.return_provider import ReturnProvider
from quant.backtest.engine import run_backtest
from quant.report.metrics import cagr, max_drawdown, calmar, turnover
from quant.report.artifacts import save_result, ensure_dir

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prices", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--params", required=True)
    ap.add_argument("--out-dir", required=True)
    return ap.parse_args()

def main():
    args = parse_args()
    uni = load_universe(args.config)
    params = load_params(args.params)

    df_long = load_prices_long(args.prices)
    prices = to_wide_adj_close(df_long)

    # Only keep relevant tickers
    needed = [t for t in uni.tickers if t in prices.columns]
    prices = prices[needed].dropna(how="all")

    # Build signals (monthly/weekly decided by params['rebalance'])
    kind_map = uni.kind_map()
    signals = build_signals(prices, kind_map, uni.regime_ticker, uni.cash_proxy, params)

    rp = ReturnProvider(
        prices_wide=prices,
        inception=uni.inception,
        annual_fees=uni.annual_fees,
        leverage_maps=uni.leverage_maps(),
    )

    equity, weights, trades = run_backtest(prices, signals, rp, params["costs"])

    metrics = {
        "CAGR": cagr(equity),
        "MDD": max_drawdown(equity),
        "Calmar": calmar(equity),
        "Turnover": turnover(weights),
        "Start": str(equity.index.min().date()),
        "End": str(equity.index.max().date()),
    }

    ensure_dir(args.out_dir)

    # save signals with parquet->csv fallback
    sig_pq = Path(args.out_dir) / "signals.parquet"
    try:
        signals.to_parquet(sig_pq, index=False)
    except Exception:
        signals.to_csv(Path(args.out_dir) / "signals.csv", index=False)

    save_result(args.out_dir, equity, weights, trades, metrics)
    print(metrics)
    print(f"saved -> {args.out_dir}")

if __name__ == "__main__":
    main()