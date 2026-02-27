from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd
import yfinance as yf
import yaml

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tickers", default="")
    ap.add_argument("--tickers-from-config", default="")
    return ap.parse_args()

def main():
    args = parse_args()
    tickers = []
    if args.tickers_from_config:
        cfg = yaml.safe_load(open(args.tickers_from_config, "r", encoding="utf-8"))
        for a in cfg["assets"]:
            tickers.append(a["ticker"])
            if "lev2" in a and a["lev2"]:
                tickers.append(a["lev2"])
            if "lev3" in a and a["lev3"]:
                tickers.append(a["lev3"])
        tickers += [cfg.get("regime_ticker", "SPY"), cfg.get("cash_proxy", "BIL")]
    if args.tickers:
        tickers += [t.strip() for t in args.tickers.split(",") if t.strip()]
    seen=set(); tlist=[]
    for t in tickers:
        if t not in seen:
            tlist.append(t); seen.add(t)
    if not tlist:
        raise SystemExit("No tickers provided. Use --tickers or --tickers-from-config")

    data = yf.download(tlist, start=args.start, auto_adjust=False, progress=False, group_by="ticker")
    rows = []
    for t in tlist:
        if isinstance(data.columns, pd.MultiIndex):
            if (t, "Adj Close") in data.columns:
                s = data[(t, "Adj Close")].dropna()
            elif (t, "Close") in data.columns:
                s = data[(t, "Close")].dropna()
            else:
                continue
        else:
            s = data["Adj Close"].dropna() if "Adj Close" in data else data["Close"].dropna()

        for dt, val in s.items():
            rows.append({"date": pd.Timestamp(dt).date(), "ticker": t, "adj_close": float(val)})

    df = pd.DataFrame(rows).dropna()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    if out.suffix.lower() == ".parquet":
        try:
            df.to_parquet(out, index=False)
        except Exception as e:
            # fallback to csv if parquet engine not available
            csv_out = out.with_suffix(".csv")
            df.to_csv(csv_out, index=False)
            print(f"parquet unavailable, saved CSV -> {csv_out}")
            return
    else:
        df.to_csv(out, index=False)
    print(f"saved {len(df):,} rows -> {out}")

if __name__ == "__main__":
    main()
