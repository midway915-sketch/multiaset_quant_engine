from __future__ import annotations
import argparse
import glob
import json
from pathlib import Path
from typing import Dict, Any, List

import numpy as np
import pandas as pd

from rolling10y_and_compare import _read_equity_csv, rolling10y_stats, _window_metrics  # re-use


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--label-and-equity",
                    action="append",
                    required=True,
                    help='repeat: "LABEL=path/to/equity.csv"')
    ap.add_argument("--out-csv", default="compare_table.csv")
    args = ap.parse_args()

    rows = []
    for item in args.label_and_equity:
        if "=" not in item:
            raise ValueError(f"Bad format: {item}. Use LABEL=equity.csv")
        label, path = item.split("=", 1)
        label = label.strip()
        path = path.strip()

        eq = _read_equity_csv(path)

        # full period
        full = _window_metrics(eq, eq["date"].min(), eq["date"].max())

        # rolling 10y stats (includes last10y)
        _, stats = rolling10y_stats(eq, window_years=10)

        rows.append({
            "label": label,
            "equity_path": path,

            "full_cagr": full["cagr"],
            "full_multiple": full["multiple"],
            "full_mdd": full["mdd"],

            "median_cagr_10y": stats.get("median_cagr"),
            "median_mdd_10y": stats.get("median_mdd"),
            "median_calmar_10y": stats.get("median_calmar"),
            "worst_cagr_10y": stats.get("worst_cagr"),
            "p10_cagr_10y": stats.get("p10_cagr"),
            "p90_cagr_10y": stats.get("p90_cagr"),
            "n_windows_10y": stats.get("n_windows"),

            "last10y_cagr": stats.get("last10y_cagr"),
            "last10y_multiple": stats.get("last10y_multiple"),
            "last10y_mdd": stats.get("last10y_mdd"),
            "last10y_start": stats.get("last10y_start"),
            "last10y_end": stats.get("last10y_end"),
        })

    outdf = pd.DataFrame(rows)
    outdf = outdf.sort_values(["median_cagr_10y", "full_cagr"], ascending=False)

    Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
    outdf.to_csv(args.out_csv, index=False)
    print("wrote:", args.out_csv)


if __name__ == "__main__":
    main()