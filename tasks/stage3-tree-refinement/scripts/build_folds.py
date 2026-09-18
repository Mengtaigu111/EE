"""Leakage-safe SHA256 document folds for reward-data construction."""

from __future__ import annotations

import hashlib
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
