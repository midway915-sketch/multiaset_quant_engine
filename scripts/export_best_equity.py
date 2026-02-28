from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from quant.data.loader import load_prices_long, to_wide_adj_close
from quant.strategy.universe import load_universe
from quant.data.return_provider import ReturnProvider
from quant.backtest.engine import run_backtest
from quant.report.artifacts import ensure_dir


def _load_params_json(path: str) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _build_signals(prices: pd.DataFrame, kind_map: dict, uni, params: dict) -> pd.DataFrame:
    """
    Compatibility wrapper:
    - prefers quant.strategy.policy.build_signals if your repo has it
    - otherwise falls back to build_monthly_signals (older engine)
    """
    from quant.strategy import policy as pol

    if hasattr(pol, "build_signals"):
        return pol.build_signals(prices, kind_map, uni.regime_ticker, uni.cash_proxy, params)
    if hasattr(pol, "build_monthly_signals"):
        return pol.build_monthly_signals(prices, kind_map, uni.regime_ticker, uni.cash_proxy, params)

    raise RuntimeError("policy.py has neither build_signals nor build_monthly_signals")


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prices", required=True, help="data/prices.csv")
    ap.add_argument("--config", required=True, help="config/universe.yml")
    ap.add_argument("--best-params-json", required=True, help="configs/best/roundX.json")
    ap.add_argument("--out-dir", required=True, help="out_compare/roundX")
    return ap.parse_args()


def main():
    args = parse_args()
    uni = load_universe(args.config)
    params = _load_params_json(args.best_params_json)

    df_long = load_prices_long(args.prices)
    prices = to_wide_adj_close(df_long)

    needed = [t for t in uni.tickers if t in prices.columns]
    prices = prices[needed].dropna(how="all")

    kind_map = uni.kind_map()
    signals = _build_signals(prices, kind_map, uni, params)

    rp = ReturnProvider(
        prices_wide=prices,
        inception=uni.inception,
        annual_fees=uni.annual_fees,
        leverage_maps=uni.leverage_maps(),
    )

    equity, weights, trades = run_backtest(prices, signals, rp, params["costs"])

    out_dir = Path(args.out_dir)
    ensure_dir(out_dir)

    # equity
    eq_df = pd.DataFrame({"date": equity.index, "equity": equity.values})
    eq_df.to_csv(out_dir / "equity.csv", index=False)

    # optional: save weights/trades (parquet preferred)
    try:
        weights.to_parquet(out_dir / "weights.parquet")
    except Exception:
        weights.to_csv(out_dir / "weights.csv")
    try:
        trades.to_parquet(out_dir / "trades.parquet")
    except Exception:
        trades.to_csv(out_dir / "trades.csv", index=False)

    print(f"saved equity -> {out_dir/'equity.csv'}")
    print(f"start={equity.index.min().date()} end={equity.index.max().date()} final_multiple={float(equity.iloc[-1]/equity.iloc[0]):.6f}")


if __name__ == "__main__":
    main()
