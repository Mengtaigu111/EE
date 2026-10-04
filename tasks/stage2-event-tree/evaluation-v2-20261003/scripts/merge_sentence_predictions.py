#!/usr/bin/env python3
"""Merge Experiment C sentence predictions into document-level aligned events."""
import argparse
from collections import Counter
import json
from pathlib import Path

from align_events import align_forest
from linearize import parse_forest


def read_jsonl(path):
    with Path(path).open(encoding='utf-8') as stream:
        return [json.loads(line) for line in stream if line.strip()]


def reparse_raw_predictions(predictions):
    reparsed = []
    for old in predictions:
        parsed, diagnostics = parse_forest(old.get('raw_output'))
        prediction = dict(old)
        prediction['parsed_events'] = parsed
        prediction['parse_diagnostics'] = diagnostics
        prediction['invalid'] = diagnostics['invalid']
        reparsed.append(prediction)
    return reparsed


def _event_key(event):
    return (event['event_type'], tuple(event['trigger']['offsets']),
            tuple(sorted((arg['argument_type'], tuple(arg['offsets'])) for arg in event['arguments'])))


def _validate_rows(rows):
    by_document = {}
    for row in rows:
        required = ('id', 'document_id', 'sentence_index', 'target_bounds', 'tokens', 'sentences')
        missing = [key for key in required if key not in row]
        if missing:
            raise ValueError(f'{row.get("id", "<unknown>")}: missing fields {missing}')
        by_document.setdefault(row['document_id'], []).append(row)
    for document_id, group in by_document.items():
        indexes = [row['sentence_index'] for row in group]
        if len(indexes) != len(set(indexes)):
            raise ValueError(f'{document_id}: duplicate sentence_index')
        expected = list(range(len(group[0]['sentences'])))
        if sorted(indexes) != expected:
            raise ValueError(f'{document_id}: sentence rows are not complete: {sorted(indexes)} vs {expected}')
        token_count = len(group[0]['tokens'])
        if any(row['tokens'] != group[0]['tokens'] or row['sentences'] != group[0]['sentences']
               for row in group):
            raise ValueError(f'{document_id}: sentence rows disagree on document context')
        if any(row['target_bounds'][0] < 0 or row['target_bounds'][1] > token_count
               or row['target_bounds'][0] >= row['target_bounds'][1] for row in group):
            raise ValueError(f'{document_id}: invalid target bounds')
    return by_document


def merge_sentence_rows(rows, predictions, deduplicate=True):
    rows_by_document = _validate_rows(rows)
    predictions_by_id = {}
    for prediction in predictions:
        prediction_id = prediction.get('id')
        if prediction_id in predictions_by_id:
            raise ValueError(f'duplicate sentence prediction: {prediction_id}')
        predictions_by_id[prediction_id] = prediction
    row_ids = {row['id'] for row in rows}
    unknown = sorted(set(predictions_by_id) - row_ids)
    if unknown:
        raise ValueError(f'unknown sentence predictions: {unknown[:3]}')
    documents = []
    for document_id, group in rows_by_document.items():
        ordered = sorted(group, key=lambda row: row['sentence_index'])
        events, scopes, raw_outputs = [], [], []
        duplicate_events = 0
        parse_totals = Counter()
        invalid = False
        hit_max_new_tokens = 0
        for row in ordered:
            prediction = predictions_by_id.get(row['id'])
            if prediction is None:
                raise ValueError(f'{document_id}: missing prediction for {row["id"]}')
            sentence_events = prediction.get('parsed_events', [])
            parse_totals.update({key: value for key, value in prediction.get('parse_diagnostics', {}).items()
                                 if key != 'invalid' and isinstance(value, int)})
            events.extend(sentence_events)
            scopes.extend([tuple(row['target_bounds'])] * len(sentence_events))
            raw_outputs.append({'id': row['id'], 'raw_output': prediction.get('raw_output', '')})
            invalid = invalid or bool(prediction.get('invalid'))
            hit_max_new_tokens += bool(prediction.get('hit_max_new_tokens'))
        first = ordered[0]
        aligned, alignment = align_forest(first['tokens'], first['sentences'], events,
                                          trigger_scopes=scopes,
                                          argument_sentence_preference=False)
        unique, seen = [], set()
        for event in sorted(aligned, key=lambda value: (value['trigger']['offsets'][0],
                                                        value['trigger']['offsets'][-1], value['event_type'])):
            key = _event_key(event)
            if deduplicate and key in seen:
                continue
            seen.add(key)
            unique.append(event)
        documents.append({
            'id': document_id,
            'document_id': document_id,
            'parsed_events': [],
            'aligned_events': unique,
            'alignment': alignment,
            'parse_diagnostics': dict(parse_totals),
            'merge_diagnostics': {'input_events': len(aligned),
                                  'duplicate_events': duplicate_events + len(aligned) - len(unique)},
            'sentence_count': len(ordered),
            'invalid': invalid,
            'hit_max_new_tokens': hit_max_new_tokens,
            'sentence_outputs': raw_outputs,
        })
    return documents


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, required=True, help='Experiment C sentence JSONL')
    parser.add_argument('--predictions', type=Path,
                        help='sentence prediction JSONL; mutually exclusive with --gold')
    parser.add_argument('--gold', action='store_true', help='parse each row completion as a gold prediction')
    parser.add_argument('--output', type=Path, required=True, help='document-level JSONL')
    args = parser.parse_args()
    if bool(args.predictions) == args.gold:
        parser.error('provide exactly one of --predictions or --gold')
    if args.output.exists():
        raise FileExistsError(args.output)
    rows = read_jsonl(args.data)
    if args.gold:
        predictions = []
        for row in rows:
            parsed, diagnostics = parse_forest(row['completion'])
            if diagnostics['invalid']:
                raise ValueError(f'{row["id"]}: invalid gold completion')
            predictions.append({'id': row['id'], 'parsed_events': parsed,
                                'invalid': False, 'hit_max_new_tokens': False})
    else:
        predictions = reparse_raw_predictions(read_jsonl(args.predictions))
    merged = merge_sentence_rows(rows, predictions, deduplicate=not args.gold)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x', encoding='utf-8') as stream:
        for row in merged:
            stream.write(json.dumps(row, ensure_ascii=False) + '\n')
    print(json.dumps({'documents': len(merged), 'output': str(args.output)}, ensure_ascii=False))


if __name__ == '__main__':
    main()
