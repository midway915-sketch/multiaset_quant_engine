from __future__ import annotations
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def equity_and_drawdown_plots(equity: pd.Series, out_dir: str):
    outp = Path(out_dir)
    outp.mkdir(parents=True, exist_ok=True)

    eq = equity.dropna().copy()
    if len(eq) < 2:
        return

    # Equity curve
    plt.figure()
    plt.plot(eq.index, eq.values)
    plt.title("Equity Curve")
    plt.xlabel("Date")
    plt.ylabel("Equity")
    plt.tight_layout()
    plt.savefig(outp / "equity_curve.png", dpi=150)
    plt.close()

    # Drawdown curve
    peak = eq.cummax()
    dd = eq / peak - 1.0
    plt.figure()
    plt.plot(dd.index, dd.values)
    plt.title("Drawdown")
    plt.xlabel("Date")
    plt.ylabel("Drawdown")
    plt.tight_layout()
    plt.savefig(outp / "drawdown.png", dpi=150)
    plt.close()
