"""Derive dev-only symmetric type confusion sets from T0 predictions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from common import read_jsonl
from diagnostics import build_confusion_sets
from sample_pool import align_text_events


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--predictions', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    data = {row['id']: row for row in read_jsonl(args.data)}
    predictions = {row['id']: row for row in read_jsonl(args.predictions)}
    if set(data) != set(predictions):
        raise ValueError('data/prediction IDs do not match')
    documents = []
    for document_id, row in data.items():
        predicted, _ = align_text_events(row, predictions[document_id]['parsed_events'])
        documents.append({'predicted': predicted, 'gold': row['gold_events']})
    result = build_confusion_sets(documents)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


if __name__ == '__main__':
    main()
