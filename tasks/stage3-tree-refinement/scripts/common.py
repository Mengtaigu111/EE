"""Shared Stage-3 helpers and the read-only bridge to Stage-2."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
from typing import Any, Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[3]
STAGE2_SCRIPTS = PROJECT_ROOT / 'tasks' / 'stage2-event-tree' / 'scripts'
if str(STAGE2_SCRIPTS) not in sys.path:
    # Keep the Stage-3 script directory ahead of this compatibility path so
    # generic module names (notably run_queue) cannot resolve to Stage-2 code.
    sys.path.append(str(STAGE2_SCRIPTS))

# These imports intentionally resolve to the unchanged Stage-2 implementation.
from align_events import align_forest  # noqa: E402
from linearize import parse_forest  # noqa: E402


def _offsets(value: Any) -> tuple[int, ...]:
    if isinstance(value, dict):
        value = value.get('offsets')
    if value is None:
        return ()
    return tuple(sorted(set(int(position) for position in value)))


def _event_type(event: dict[str, Any]) -> str:
    value = event.get('type', event.get('event_type'))
    if not value:
        raise ValueError(f'candidate event has no type: {event!r}')
    return str(value)


def _trigger_offsets(event: dict[str, Any]) -> tuple[int, ...]:
    value = event.get('trigger_offsets', event.get('trigger'))
    offsets = _offsets(value)
    if not offsets:
        raise ValueError(f'candidate event has no trigger offsets: {event!r}')
    return offsets


def _arguments(event: dict[str, Any]) -> Iterable[dict[str, Any]]:
    return event.get('args', event.get('arguments', ()))


def _argument_role(argument: dict[str, Any]) -> str:
    value = argument.get('role', argument.get('argument_type'))
    if not value:
        raise ValueError(f'candidate argument has no role: {argument!r}')
    return str(value)


def _argument_offsets(argument: dict[str, Any]) -> tuple[int, ...]:
    offsets = _offsets(argument)
    if not offsets:
        raise ValueError(f'candidate argument has no offsets: {argument!r}')
    return offsets


def trigger_key(event: dict[str, Any]) -> tuple[int, ...]:
    """Return the canonical token-offset identity of an event trigger."""
    return _trigger_offsets(event)


def event_key(event: dict[str, Any]) -> tuple[str, tuple[int, ...], frozenset[tuple[str, tuple[int, ...]]]]:
    """Return the spec-defined identity, ignoring duplicate arguments."""
    arguments = frozenset((_argument_role(argument), _argument_offsets(argument))
                          for argument in _arguments(event))
    return _event_type(event), _trigger_offsets(event), arguments


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows = []
    with Path(path).open(encoding='utf-8') as stream:
        for line_number, line in enumerate(stream, 1):
            if line.strip():
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    raise ValueError(f'{path}:{line_number}: invalid JSON') from exc
    return rows


def write_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8') as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n')


def atomic_write_json(path: str | Path, value: Any) -> None:
    """Publish one complete JSON document, leaving no temporary file behind."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(prefix=f'.{path.name}.', suffix='.tmp',
                                           dir=path.parent)
    try:
        with os.fdopen(fd, 'w', encoding='utf-8') as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write('\n')
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_name, path)
    finally:
        if os.path.exists(temporary_name):
            os.unlink(temporary_name)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()
