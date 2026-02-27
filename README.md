# Multi-Asset Leveraged Quant Engine (v2)

This is a modular research engine designed for:
- Multi-asset rotation (top-N) using risk-adjusted momentum
- SPY regime filter (MA200) -> risk-off to a cash proxy (BIL by default)
- Conditional leverage gears (1x/2x/3x) for equity assets
- Synthetic leverage *before* leveraged ETFs existed, and auto-switch to real leveraged ETF returns after inception
- Transaction cost model (buy/sell) + optional annual fee (daily) for leveraged products
- Grid + walk-forward research workflow (GitHub Actions ready)

## Quick start (local)
1) Install deps:
```bash
pip install -e .
```

2) Download data:
```bash
python scripts/download_prices.py --start 2000-01-01 --tickers-from-config config/universe.yml --out data/prices.parquet
```

3) Run a single backtest:
```bash
python scripts/run_single.py --prices data/prices.parquet --config config/universe.yml --params config/default_params.yml --out-dir out_single
```

4) Run a grid + walk-forward:
```bash
python scripts/run_grid.py --prices data/prices.parquet --config config/universe.yml --grid config/grid.yml --out-dir out_grid
```

## Data format
`data/prices.parquet` is long format:
- date (YYYY-MM-DD)
- ticker
- adj_close (required)
- close (optional)
- volume (optional)
