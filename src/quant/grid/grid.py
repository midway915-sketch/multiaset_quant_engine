from __future__ import annotations
from typing import Dict, Any, List
import copy
import hashlib
import json
from quant.core.utils import dict_product, deep_set

def make_param_sets(base_params: Dict[str, Any], grid: Dict[str, List[Any]]) -> List[Dict[str, Any]]:
    out = []
    for overrides in dict_product(grid):
        p = copy.deepcopy(base_params)
        for k, v in overrides.items():
            deep_set(p, k, v)
        out.append(p)
    return out

def param_id(params: Dict[str, Any]) -> str:
    s = json.dumps(params, sort_keys=True)
    return hashlib.md5(s.encode("utf-8")).hexdigest()[:10]
