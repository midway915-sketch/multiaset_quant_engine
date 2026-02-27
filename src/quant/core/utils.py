from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, Iterable, Tuple
import copy

def deep_get(d: Dict[str, Any], path: str):
    cur = d
    for p in path.split("."):
        cur = cur[p]
    return cur

def deep_set(d: Dict[str, Any], path: str, value: Any):
    cur = d
    parts = path.split(".")
    for p in parts[:-1]:
        cur = cur[p]
    cur[parts[-1]] = value

def dict_product(grid: Dict[str, Iterable[Any]]):
    # grid: {"a.b": [1,2], "c": [3,4]}
    items = list(grid.items())
    if not items:
        yield {}
        return
    def rec(i: int, acc: Dict[str, Any]):
        if i == len(items):
            yield acc
            return
        k, vals = items[i]
        for v in vals:
            nxt = copy.deepcopy(acc)
            nxt[k] = v
            yield from rec(i + 1, nxt)
    yield from rec(0, {})
