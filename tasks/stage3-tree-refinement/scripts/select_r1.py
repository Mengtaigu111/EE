"""R1 type-competition selection primitives."""

from __future__ import annotations

from typing import Any, Iterable

from common import event_key, trigger_key


def select_type_competitor(candidates: Iterable[dict[str, Any]]) -> dict[str, Any]:
    candidates = list(candidates)
    if not candidates:
        raise ValueError('cannot select from an empty competitor set')
    return sorted(candidates, key=lambda row: (
        -float(row.get('reward_score', row.get('score', 0.0))),
        row['type'], repr(event_key(row)),
    ))[0]


def replace_by_trigger(base_events: Iterable[dict[str, Any]],
                       competitors_by_trigger: dict[tuple[int, ...], list[dict[str, Any]]]) -> list[dict[str, Any]]:
    result = []
    for base in base_events:
        competitors = competitors_by_trigger.get(trigger_key(base))
        result.append(select_type_competitor(competitors) if competitors else base)
    return sorted(result, key=lambda row: (trigger_key(row), row['type'], repr(event_key(row))))
