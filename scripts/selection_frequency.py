from __future__ import annotations
import argparse
import ast
import json
from pathlib import Path
from typing import Any, List

import pandas as pd


def _parse_assets(x: Any) -> List[str]:
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return []
    if isinstance(x, list):
        return [str(a) for a in x]
    s = str(x).strip()

    # try json
    try:
        v = json.loads(s)
        if isinstance(v, list):
            return [str(a) for a in v]
    except Exception:
        pass

    # try python literal
    try:
        v = ast.literal_eval(s)
        if isinstance(v, list):
            return [str(a) for a in v]
    except Exception:
        pass

    # fallback: single ticker
    return [s]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-csv", required=True, help="csv containing 'assets' column")
    ap.add_argument("--assets-col", default="assets")
    ap.add_argument("--out-csv", default="selection_frequency.csv")
    args = ap.parse_args()

    df = pd.read_csv(args.input_csv)
    if args.assets_col not in df.columns:
        raise ValueError(f"Missing column '{args.assets_col}'. Columns={list(df.columns)}")

    all_assets = []
    for x in df[args.assets_col].values:
        all_assets.extend(_parse_assets(x))

    s = pd.Series(all_assets, dtype="string")
    freq = s.value_counts(dropna=True).rename_axis("asset").reset_index(name="count")
    freq["share"] = freq["count"] / float(freq["count"].sum()) if len(freq) else 0.0

    Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
    freq.to_csv(args.out_csv, index=False)
    print("wrote:", args.out_csv)


if __name__ == "__main__":
    main()