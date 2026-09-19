"""R1 type-competition selection primitives."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable

from common import event_key, read_jsonl, trigger_key
from sample_pool import align_text_events


def select_type_competitor(candidates: Iterable[dict[str, Any]]) -> dict[str, Any]:
    candidates = list(candidates)
    if not candidates:
        raise ValueError('cannot select from an empty competitor set')
    return sorted(candidates, key=lambda row: (
        -float(row.get('reward_score', row.get('score', 0.0))),
        row['type'], repr(event_key(row)),
    ))[0]


def select_oracle_competitor(candidates: Iterable[dict[str, Any]]) -> dict[str, Any]:
    """Gold-labelled upper bound for a competitor set; never use at test time."""
    candidates = list(candidates)
    if not candidates:
        raise ValueError('cannot select from an empty competitor set')
    return sorted(candidates, key=lambda row: (
        -int(row.get('label', {}).get('type_ok', 0)),
        -float(row.get('label', {}).get('event_score', 0.0)),
        -float(row.get('label', {}).get('arg_r', 0.0)),
        repr(event_key(row)),
    ))[0]


def replace_by_trigger(base_events: Iterable[dict[str, Any]],
                       competitors_by_trigger: dict[tuple[int, ...], list[dict[str, Any]]]) -> list[dict[str, Any]]:
    result = []
    for base in base_events:
        competitors = competitors_by_trigger.get(trigger_key(base))
        result.append(select_type_competitor(competitors) if competitors else base)
    return sorted(result, key=lambda row: (trigger_key(row), row['type'], repr(event_key(row))))


def select_document_events(base_events: Iterable[dict[str, Any]],
                           competitor_rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    by_index: dict[int, list[dict[str, Any]]] = {}
    for row in competitor_rows:
        candidate = row.get('candidate')
        if candidate is not None:
            by_index.setdefault(int(row['event_index']), []).append(candidate)
    selected = [select_type_competitor(by_index[index]) if index in by_index else event
                for index, event in enumerate(base_events)]
    return selected


def select_text_document_events(base_events: Iterable[dict[str, Any]],
                                competitor_rows: Iterable[dict[str, Any]],
                                score_key: str = 'score_sum') -> list[dict[str, Any]]:
    """Select among D2 text subtrees before document-level offset alignment."""
    by_index: dict[int, list[dict[str, Any]]] = {}
    for row in competitor_rows:
        if row.get('candidate') is not None and row.get('closed', False):
            by_index.setdefault(int(row['event_index']), []).append(row)
    selected = []
    for index, event in enumerate(base_events):
        options = by_index.get(index)
        if not options:
            selected.append(event)
            continue
        best = sorted(options, key=lambda row: (-float(row.get(score_key, float('-inf'))),
                                                 row['candidate']['type'], repr(row['candidate'])))[0]
        selected.append(best['candidate'])
    return selected


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--predictions', type=Path, required=True)
    parser.add_argument('--competitors', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--score', choices=('score_sum', 'score_mean'), default='score_sum')
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    data = {row['id']: row for row in read_jsonl(args.data)}
    predictions = {row['id']: row for row in read_jsonl(args.predictions)}
    competitors = {row['id']: row for row in read_jsonl(args.competitors)}
    if set(data) != set(predictions) or set(data) != set(competitors):
        raise ValueError('data, predictions and competitor IDs do not match')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x', encoding='utf-8') as stream:
        for document_id, row in data.items():
            selected = select_text_document_events(
                predictions[document_id]['parsed_events'],
                competitors[document_id]['candidates'], args.score)
            aligned, alignment = align_text_events(row, selected)
            stream.write(json.dumps({'id': document_id, 'split': row.get('split'),
                                     'aligned_events': aligned, 'alignment': alignment,
                                     'mode': f'R1-{args.score}'},
                                    ensure_ascii=False, allow_nan=False) + '\n')


if __name__ == '__main__':
    main()
