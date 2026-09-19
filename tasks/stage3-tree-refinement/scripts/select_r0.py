"""R0 candidate-pool filtering and dev-only threshold grid."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable

from common import event_key, read_jsonl, trigger_key
from sample_pool import candidate_to_aligned_event


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
    # A trigger may legitimately instantiate two event types, but retaining
    # every high-scoring type turns one local ambiguity into an FP burst.  v1
    # therefore keeps the two highest-scoring distinct types per trigger.
    by_trigger: dict[tuple[int, ...], list[dict[str, Any]]] = {}
    for candidate in by_trigger_type.values():
        by_trigger.setdefault(trigger_key(candidate), []).append(candidate)
    selected = []
    for trigger in sorted(by_trigger):
        selected.extend(sorted(by_trigger[trigger], key=_rank)[:2])
    return sorted(selected, key=_rank)


def _rank(row: dict[str, Any]) -> tuple[Any, ...]:
    return (trigger_key(row), -float(row.get('reward_score', row.get('score', 0.0))),
            row['type'], repr(event_key(row)))


def choose_tau(candidates: Iterable[dict[str, Any]], metric_fn: Any) -> tuple[float, list[dict[str, Any]]]:
    """Select tau on dev only, maximizing AC with a stable low-tau tie break."""
    candidates = list(candidates)
    curve = []
    for tau in tau_grid():
        selected = filter_candidates(candidates, tau)
        metrics = dict(metric_fn(selected))
        curve.append({'tau': tau, 'events': len(selected), **metrics})
    best = max(curve, key=lambda row: (float(row.get('AC_f1', row.get('AC', 0.0))), -row['tau']))
    return float(best['tau']), curve


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--pool', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--tau', type=float, required=True)
    parser.add_argument('--mode', choices=('dev', 'test'), default='test')
    parser.add_argument('--tau-config', type=Path,
                        help='Frozen dev config JSON required in test mode')
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if args.mode == 'test' and args.tau_config is not None:
        config = json.loads(args.tau_config.read_text(encoding='utf-8'))
        if round(float(config.get('tau')), 2) != round(args.tau, 2):
            raise ValueError('test tau does not match frozen dev tau config')
    data = {row['id']: row for row in read_jsonl(args.data)}
    pools = {row['document_id']: row for row in read_jsonl(args.pool)}
    if set(data) != set(pools):
        raise ValueError('data/pool IDs do not match')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x', encoding='utf-8') as stream:
        for document_id, row in data.items():
            selected = filter_candidates(pools[document_id]['candidates'], args.tau)
            aligned = [candidate_to_aligned_event(candidate, row['tokens']) for candidate in selected]
            stream.write(json.dumps({'id': document_id, 'split': row.get('split'),
                                     'aligned_events': aligned, 'tau': args.tau,
                                     'mode': 'R0'}, ensure_ascii=False, allow_nan=False) + '\n')


if __name__ == '__main__':
    main()
