from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Optional
import yaml

from quant.data.return_provider import LeverMap

@dataclass(frozen=True)
class Asset:
    ticker: str
    kind: str  # equity, bond, alt, fx, cash
    lev2: Optional[str] = None
    lev3: Optional[str] = None

@dataclass
class Universe:
    regime_ticker: str
    cash_proxy: str
    assets: List[Asset]
    inception: Dict[str, str]
    annual_fees: Dict[str, float]

    @property
    def tickers(self) -> List[str]:
        ts = []
        for a in self.assets:
            ts.append(a.ticker)
            if a.lev2: ts.append(a.lev2)
            if a.lev3: ts.append(a.lev3)
        # dedupe keep order
        seen = set()
        out = []
        for t in ts:
            if t not in seen:
                out.append(t); seen.add(t)
        return out

    def leverage_maps(self) -> Dict[str, LeverMap]:
        m = {}
        for a in self.assets:
            m[a.ticker] = LeverMap(base=a.ticker, lev2=a.lev2, lev3=a.lev3)
        return m

    def kind_map(self) -> Dict[str, str]:
        return {a.ticker: a.kind for a in self.assets}

def load_universe(path: str) -> Universe:
    cfg = yaml.safe_load(open(path, "r", encoding="utf-8"))
    assets = [Asset(**a) for a in cfg["assets"]]
    return Universe(
        regime_ticker=cfg["regime_ticker"],
        cash_proxy=cfg["cash_proxy"],
        assets=assets,
        inception=cfg.get("inception", {}),
        annual_fees=cfg.get("annual_fees", {"1x":0.0,"2x":0.0095,"3x":0.0095}),
    )
