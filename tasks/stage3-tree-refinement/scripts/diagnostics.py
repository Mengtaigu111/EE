"""Candidate labels and diagnostics relative to the frozen T0 prediction."""

from __future__ import annotations

from collections import Counter
from typing import Any, Iterable

from common import _argument_offsets, _argument_role, _arguments, _event_type, event_key, trigger_key


def _argument_set(event: dict[str, Any]) -> frozenset[tuple[str, tuple[int, ...]]]:
    return frozenset((_argument_role(argument), _argument_offsets(argument))
                     for argument in _arguments(event))


def _f1(predicted: set[Any], gold: set[Any]) -> tuple[float, float, float]:
    if not predicted and not gold:
        return 1.0, 1.0, 1.0
    overlap = len(predicted & gold)
    precision = overlap / len(predicted) if predicted else 0.0
    recall = overlap / len(gold) if gold else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return precision, recall, f1


def _gold_event_type(event: dict[str, Any]) -> str:
    return _event_type(event)


def _gold_index(gold_events: Iterable[dict[str, Any]]) -> tuple[set[tuple[int, ...]], set[tuple[str, tuple[int, ...]]], dict[tuple[str, tuple[int, ...]], list[dict[str, Any]]]]:
    trigger_keys: set[tuple[int, ...]] = set()
    type_keys: set[tuple[str, tuple[int, ...]]] = set()
    by_type_trigger: dict[tuple[str, tuple[int, ...]], list[dict[str, Any]]] = {}
    for event in gold_events:
        event_type = _gold_event_type(event)
        trigger = trigger_key(event)
        trigger_keys.add(trigger)
        type_key = (event_type, trigger)
        type_keys.add(type_key)
        by_type_trigger.setdefault(type_key, []).append(event)
    return trigger_keys, type_keys, by_type_trigger


def label_candidate(candidate: dict[str, Any], gold_events: Iterable[dict[str, Any]]) -> dict[str, float | int]:
    """Apply the spec's exists/type_ok/argument labels to one aligned candidate."""
    trigger_keys, type_keys, by_type_trigger = _gold_index(gold_events)
    trigger = trigger_key(candidate)
    event_type = _event_type(candidate)
    exists = int(trigger in trigger_keys)
    type_ok = int((event_type, trigger) in type_keys)
    if not type_ok:
        return {'exists': exists, 'type_ok': 0, 'arg_p': 0.0, 'arg_r': 0.0,
                'arg_f1': 0.0, 'event_score': 0.0}

    predicted_args = set(_argument_set(candidate))
    best = max(
        (_f1(predicted_args, set(_argument_set(event)))
         for event in by_type_trigger[(event_type, trigger)]),
        key=lambda values: (values[2], values[1], values[0]),
    )
    precision, recall, f1 = best
    return {'exists': exists, 'type_ok': 1, 'arg_p': precision, 'arg_r': recall,
            'arg_f1': f1, 'event_score': f1}


def build_confusion_sets(dev_documents: Iterable[dict[str, Any]], top_k: int = 3) -> dict[str, list[str]]:
    """Build symmetric type confusion lists from trigger-correct dev events."""
    if top_k < 1:
        raise ValueError('top_k must be positive')
    counts: dict[str, Counter[str]] = {}
    for document in dev_documents:
        gold_by_trigger: dict[tuple[int, ...], set[str]] = {}
        for gold in document.get('gold', []):
            gold_by_trigger.setdefault(trigger_key(gold), set()).add(_gold_event_type(gold))
        for predicted in document.get('predicted', []):
            gold_types = gold_by_trigger.get(trigger_key(predicted), set())
            predicted_type = _event_type(predicted)
            for gold_type in gold_types:
                if gold_type == predicted_type:
                    continue
                counts.setdefault(gold_type, Counter())[predicted_type] += 1
                counts.setdefault(predicted_type, Counter())[gold_type] += 1
    return {
        event_type: [other for other, _ in sorted(
            values.items(), key=lambda item: (-item[1], item[0]))[:top_k]]
        for event_type, values in sorted(counts.items())
    }


def relative_diagnostics(initial: Iterable[dict[str, Any]], result: Iterable[dict[str, Any]],
                         gold_events: Iterable[dict[str, Any]]) -> dict[str, Any]:
    """Count T0-relative event changes using candidate identities."""
    gold_events = list(gold_events)
    initial = list(initial)
    result = list(result)
    initial_keys = {event_key(event) for event in initial}
    result_keys = {event_key(event) for event in result}
    initial_by_key = {event_key(event): event for event in initial}
    result_by_key = {event_key(event): event for event in result}

    def is_type_ok(event: dict[str, Any]) -> bool:
        return bool(label_candidate(event, gold_events)['type_ok'])

    initial_types = {(trigger_key(event), _event_type(event)): event for event in initial}
    result_types = {(trigger_key(event), _event_type(event)): event for event in result}
    type_corrected = sum(
        not is_type_ok(event) and any(
            trigger_key(other) == trigger_key(event) and is_type_ok(other)
            for other in result
        ) for event in initial
    )
    type_corrupted = sum(
        is_type_ok(event) and not any(
            trigger_key(other) == trigger_key(event) and is_type_ok(other)
            for other in result
        ) for event in initial
    )
    initial_trigger_keys = {trigger_key(event) for event in initial}
    recovered = sum(
        trigger_key(event) not in initial_trigger_keys and label_candidate(event, gold_events)['exists']
        for event in result
    )
    added_fp = sum(
        trigger_key(event) not in initial_trigger_keys and not label_candidate(event, gold_events)['exists']
        for event in result
    )
    removed_tp = sum(label_candidate(event, gold_events)['exists']
                     for key, event in initial_by_key.items() if key not in result_keys)
    removed_fp = sum(not label_candidate(event, gold_events)['exists']
                     for key, event in initial_by_key.items() if key not in result_keys)
    pairs = Counter()
    for event in initial:
        pairs[(_gold_type_for_trigger(event, gold_events), _event_type(event), 'initial')] += 1
    for event in result:
        pairs[(_gold_type_for_trigger(event, gold_events), _event_type(event), 'result')] += 1
    return {
        'type_corrected': int(type_corrected),
        'type_corrupted': int(type_corrupted),
        'events_recovered': int(recovered),
        'events_added_fp': int(added_fp),
        'events_removed_tp': int(removed_tp),
        'events_removed_fp': int(removed_fp),
        'per_pair': [
            {'gold_type': gold_type, 'pred_type': pred_type, 'stage': stage, 'count': count}
            for (gold_type, pred_type, stage), count in sorted(pairs.items())
        ],
    }


def _gold_type_for_trigger(event: dict[str, Any], gold_events: Iterable[dict[str, Any]]) -> str:
    trigger = trigger_key(event)
    types = sorted({_gold_event_type(gold) for gold in gold_events if trigger_key(gold) == trigger})
    return types[0] if types else '<NONE>'
