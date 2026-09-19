"""Build deterministic reward candidates and bounded synthetic negatives."""

from __future__ import annotations

import argparse
from copy import deepcopy
import json
import random
from pathlib import Path
from typing import Any, Iterable

from diagnostics import label_candidate
from score_candidates import build_reward_row
from sample_pool import candidate_from_aligned_event
from common import read_jsonl, write_jsonl


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
    raw_args = result.get('args', result.get('arguments', []))
    normalized = []
    for argument in raw_args:
        role = argument.get('role', argument.get('argument_type'))
        if role is None:
            continue
        normalized.append({
            'role': role,
            'offsets': list(argument.get('offsets', [])),
            'text': argument.get('text', ' '.join(argument.get('tokens', []))),
        })
    allowed = schema.get(event_type, set())
    result['args'] = [argument for argument in normalized if argument['role'] in allowed]
    result.pop('arguments', None)
    if drop_argument and result['args']:
        random.Random(seed).shuffle(result['args'])
        result['args'].pop()
    result['source'] = 'synthetic'
    result['synthetic'] = True
    return result


def build_synthetic_negatives(event: dict[str, Any], confusable_types: list[str],
                              schema: dict[str, set[str]], seed: int = 42) -> list[dict[str, Any]]:
    """Create the two spec-defined negatives for one gold event.

    1. Wrong type: swap to the first frozen confusable type, keep every
       schema-compatible argument (teaches the type head).
    2. Missing branch: keep the GOLD type and drop one argument branch
       (teaches the event-score head that an incomplete tree scores < 1).
    """
    if not confusable_types:
        return []
    target = confusable_types[0]
    gold_type = event.get('type', event.get('event_type'))
    return [
        synthetic_negative(event, target, schema, seed=seed, drop_argument=False),
        synthetic_negative(event, gold_type, schema, seed=seed + 1, drop_argument=True),
    ]


def build_reward_rows(pool_rows: Iterable[dict[str, Any]], data_rows: Iterable[dict[str, Any]],
                      confusion_sets: dict[str, list[str]], schema: dict[str, set[str]],
                      seed: int = 42) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    data_by_id = {row['id']: row for row in data_rows}
    real_rows, synthetic_rows = [], []
    for pool in pool_rows:
        data_row = data_by_id[pool['document_id']]
        c0 = next((source for source in pool.get('sources', []) if source.get('source') == 'c0'), None)
        if c0 is None:
            raise ValueError(f'{pool["document_id"]}: missing c0 greedy forest')
        forest = c0.get('events', [])
        for candidate in pool.get('candidates', []):
            candidate = deepcopy(candidate)
            candidate['source'] = candidate.get('source', 'real')
            real_rows.append(build_reward_row(data_row, forest, candidate))
        for event_index, gold in enumerate(data_row.get('gold_events', [])):
            gold_candidate = candidate_from_aligned_event(gold)
            options = confusion_sets.get(gold['event_type'], [])
            if not options:
                continue
            for negative_index, negative in enumerate(build_synthetic_negatives(
                    gold_candidate, options, schema, seed=seed + event_index * 2)):
                negative['id'] = f'{pool["document_id"]}:synthetic:{event_index}:{negative_index}'
                negative['label'] = label_candidate(negative, data_row.get('gold_events', []))
                synthetic_rows.append(build_reward_row(data_row, forest, negative))
    rows = cap_synthetic_rows(real_rows, synthetic_rows)
    manifest = {
        'real_rows': len(real_rows), 'synthetic_rows_available': len(synthetic_rows),
        'synthetic_rows_used': len(rows) - len(real_rows),
        'synthetic_ratio': (len(rows) - len(real_rows)) / len(real_rows) if real_rows else 0.0,
        'seed': seed,
    }
    return rows, manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pool', type=Path, nargs='+', required=True)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--confusion-sets', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    from ontology import ROLE_ORDER

    pools = [row for path in args.pool for row in read_jsonl(path)]
    data_rows = read_jsonl(args.data)
    confusion = json.loads(args.confusion_sets.read_text(encoding='utf-8'))
    schema = {event_type: set(roles) for event_type, roles in ROLE_ORDER.items()}
    rows, manifest_data = build_reward_rows(pools, data_rows, confusion, schema, args.seed)
    write_jsonl(args.output, rows)
    args.output.with_suffix('.manifest.json').write_text(
        json.dumps(manifest_data, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


if __name__ == '__main__':
    main()
