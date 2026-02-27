from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd

from quant.report.generate import generate_report
from quant.report.plots import equity_and_drawdown_plots

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", required=True, help="Directory that contains equity.* / weights.* / trades.*")
    return ap.parse_args()

def read_df(out_dir: Path, stem: str) -> pd.DataFrame:
    pq = out_dir / f"{stem}.parquet"
    csv = out_dir / f"{stem}.csv"
    if pq.exists():
        return pd.read_parquet(pq)
    if csv.exists():
        return pd.read_csv(csv, index_col=0 if stem in ("equity", "weights") else None, parse_dates=True)
    raise FileNotFoundError(f"missing {stem}.parquet or {stem}.csv")

def main():
    args = parse_args()
    out_dir = Path(args.out_dir)

    equity_df = read_df(out_dir, "equity")
    equity = equity_df["equity"]
    weights = read_df(out_dir, "weights")
    try:
        trades = read_df(out_dir, "trades")
    except FileNotFoundError:
        trades = pd.DataFrame()

    generate_report(str(out_dir), equity, weights, trades)
    equity_and_drawdown_plots(equity, str(out_dir))
    print(f"Report saved to: {out_dir}")

if __name__ == "__main__":
    main()
