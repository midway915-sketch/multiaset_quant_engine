from __future__ import annotations
from pathlib import Path
import json
import pandas as pd

def ensure_dir(p: str):
    Path(p).mkdir(parents=True, exist_ok=True)

def save_result(out_dir: str, equity: pd.Series, weights: pd.DataFrame, trades: pd.DataFrame, metrics: dict):
    ensure_dir(out_dir)
    # Prefer parquet, but fall back to CSV if parquet engine is missing
    outp = Path(out_dir)

    def try_parquet(df: pd.DataFrame, path: Path):
        try:
            df.to_parquet(path, index=True)
            return True
        except Exception:
            return False

    eq_df = equity.to_frame("equity")
    if not try_parquet(eq_df, outp / "equity.parquet"):
        eq_df.to_csv(outp / "equity.csv", index=True)

    if not try_parquet(weights, outp / "weights.parquet"):
        weights.to_csv(outp / "weights.csv", index=True)

    if trades is not None and len(trades):
        if not try_parquet(trades, outp / "trades.parquet"):
            trades.to_csv(outp / "trades.csv", index=False)

    outp.joinpath("metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
