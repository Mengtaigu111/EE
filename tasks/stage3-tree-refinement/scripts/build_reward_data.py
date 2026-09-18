"""Build deterministic reward candidates and bounded synthetic negatives."""

from __future__ import annotations

from copy import deepcopy
import random
from typing import Any, Iterable


def cap_synthetic_rows(real_rows: Iterable[dict[str, Any]], synthetic_rows: Iterable[dict[str, Any]],
                       max_ratio: float = 0.30) -> list[dict[str, Any]]:
    real = [dict(row) for row in real_rows]
    synthetic = sorted((dict(row) for row in synthetic_rows), key=lambda row: str(row.get('id', '')))
    if not 0 <= max_ratio <= 1:
        raise ValueError('max_ratio must be in [0, 1]')
    allowed = int(len(real) * max_ratio)
    return real + synthetic[:allowed]


def synthetic_negative(event: dict[str, Any], event_type: str,
                       schema: dict[str, set[str]], seed: int = 42,
                       drop_argument: bool = True) -> dict[str, Any]:
    """Swap to a confusable type and optionally remove one argument branch."""
    result = deepcopy(event)
    result['type'] = event_type
    allowed = schema.get(event_type, set())
    result['args'] = [argument for argument in result.get('args', [])
                      if argument.get('role') in allowed]
    if drop_argument and result['args']:
        random.Random(seed).shuffle(result['args'])
        result['args'].pop()
    result['source'] = 'synthetic'
    result['synthetic'] = True
    return result
