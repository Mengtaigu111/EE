"""R0 candidate-pool filtering and dev-only threshold grid."""

from __future__ import annotations

from typing import Any, Iterable

from common import event_key, trigger_key


def tau_grid() -> list[float]:
    return [round(0.1 + 0.05 * index, 2) for index in range(17)]


def filter_candidates(candidates: Iterable[dict[str, Any]], tau: float) -> list[dict[str, Any]]:
    if not 0 <= tau <= 1:
        raise ValueError('tau must be in [0, 1]')
    eligible = [row for row in candidates if float(row.get('reward_score', row.get('score', 0.0))) >= tau]
    by_trigger_type: dict[tuple[tuple[int, ...], str], dict[str, Any]] = {}
    for candidate in eligible:
        key = (trigger_key(candidate), candidate['type'])
        previous = by_trigger_type.get(key)
        if previous is None or _rank(candidate) < _rank(previous):
            by_trigger_type[key] = candidate
    return sorted(by_trigger_type.values(), key=_rank)


def _rank(row: dict[str, Any]) -> tuple[Any, ...]:
    return (trigger_key(row), -float(row.get('reward_score', row.get('score', 0.0))),
            row['type'], repr(event_key(row)))
