"""Leakage-safe SHA256 document folds for reward-data construction."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from collections import defaultdict
from typing import Any, Iterable


def assign_fold(document_id: str) -> int:
    digest = hashlib.sha256(str(document_id).encode('utf-8')).hexdigest()[:8]
    return int(digest, 16) % 2


def split_rows(rows: Iterable[dict[str, Any]]) -> dict[int, list[dict[str, Any]]]:
    folds = defaultdict(list)
    documents: dict[str, int] = {}
    for row in rows:
        document_id = str(row.get('document_id', row.get('id')))
        if not document_id or document_id == 'None':
            raise ValueError('every row needs document_id or id')
        fold = assign_fold(document_id)
        previous = documents.setdefault(document_id, fold)
        if previous != fold:
            raise AssertionError(f'document assigned to two folds: {document_id}')
        folds[fold].append(dict(row, fold=fold))
    return {0: folds[0], 1: folds[1]}


def manifest(rows: Iterable[dict[str, Any]]) -> dict[str, Any]:
    folds = split_rows(rows)
    return {
        'folds': {
            str(index): {
                'documents': sorted({row.get('document_id', row.get('id')) for row in values}),
                'rows': len(values),
            }
            for index, values in folds.items()
        }
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    rows = [json.loads(line) for line in args.data.read_text(encoding='utf-8').splitlines() if line.strip()]
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest(rows), ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


if __name__ == '__main__':
    main()
