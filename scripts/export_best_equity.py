from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from quant.data.loader import load_prices_long, to_wide_adj_close
from quant.strategy.universe import load_universe
from quant.strategy.policy import build_signals
from quant.data.return_provider import ReturnProvider
from quant.backtest.engine import run_backtest
from quant.report.artifacts import ensure_dir


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prices", required=True, help="e.g. data/prices.csv")
    ap.add_argument("--config", required=True, help="e.g. config/universe.yml")
    ap.add_argument("--best-params-json", required=True, help="e.g. out_grid_xxx/best_params.json")
    ap.add_argument("--out-dir", required=True, help="e.g. out_grid_xxx/best_full")
    return ap.parse_args()


def _save_series_csv(s: pd.Series, path: Path, name: str):
    df = pd.DataFrame({"date": s.index, name: s.values})
    df.to_csv(path, index=False)


def main():
    args = parse_args()

    uni = load_universe(args.config)
    best = json.loads(Path(args.best_params_json).read_text(encoding="utf-8"))

    df_long = load_prices_long(args.prices)
    prices = to_wide_adj_close(df_long)

    needed = [t for t in uni.tickers if t in prices.columns]
    prices = prices[needed].dropna(how="all")

    kind_map = uni.kind_map()
    signals = build_signals(prices, kind_map, uni.regime_ticker, uni.cash_proxy, best)

    rp = ReturnProvider(
        prices_wide=prices,
        inception=uni.inception,
        annual_fees=uni.annual_fees,
        leverage_maps=uni.leverage_maps(),
    )

    equity, weights, trades = run_backtest(prices, signals, rp, best["costs"])

    out_dir = Path(args.out_dir)
    ensure_dir(out_dir)

    # equity
    _save_series_csv(equity, out_dir / "equity.csv", "equity")

    # weights (wide)
    try:
        weights.to_parquet(out_dir / "weights.parquet")
    except Exception:
        weights.to_csv(out_dir / "weights.csv")

    # trades
    try:
        trades.to_parquet(out_dir / "trades.parquet")
    except Exception:
        trades.to_csv(out_dir / "trades.csv", index=False)

    # signals
    try:
        signals.to_parquet(out_dir / "signals.parquet", index=False)
    except Exception:
        signals.to_csv(out_dir / "signals.csv", index=False)

    print(f"Saved best full-period equity -> {out_dir / 'equity.csv'}")
    print(f"Equity range: {equity.index.min().date()} ~ {equity.index.max().date()}")
    print(f"Final multiple: {float(equity.iloc[-1] / equity.iloc[0]):.4f}")


if __name__ == "__main__":
    main()