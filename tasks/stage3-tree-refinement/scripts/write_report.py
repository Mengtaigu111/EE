"""Render a Stage-3 report from frozen JSON artifacts only.

The report writer is intentionally a reader, not an evaluator: it never calls
the official scorer, opens JSONL predictions, or invents a metric when the
corresponding JSON artifact is absent. This keeps a report reproducible and
makes an incomplete queue visible instead of silently presenting a partial
result as a final experiment.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Iterable


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _inside(root: Path, path: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
    except ValueError:
        return False
    return True


def collect_artifacts(run_dir: str | Path) -> list[dict[str, Any]]:
    """Load every regular JSON artifact below *run_dir* with provenance."""
    root = Path(run_dir).resolve()
    if not root.is_dir():
        raise ValueError(f'run directory does not exist: {run_dir}')
    artifacts = []
    for path in sorted(root.rglob('*.json')):
        if path.is_symlink() and not _inside(root, path):
            raise ValueError(f'artifact escapes run directory: {path}')
        if not path.is_file():
            continue
        try:
            payload = json.loads(path.read_text(encoding='utf-8'))
        except json.JSONDecodeError as exc:
            raise ValueError(f'invalid JSON artifact: {path}') from exc
        artifacts.append({
            'path': str(path.relative_to(root)),
            'bytes': path.stat().st_size,
            'sha256': _sha256(path),
            'json': payload,
        })
    return artifacts


def collect_json(run_dir: str | Path) -> dict[str, Any]:
    """Compatibility view used by the original CPU test and callers."""
    return {item['path']: item['json'] for item in collect_artifacts(run_dir)}


def _scalar(value: Any) -> bool:
    return value is None or isinstance(value, (bool, int, float, str))


def _scalar_leaves(value: Any, prefix: str = '') -> Iterable[tuple[str, Any]]:
    if _scalar(value):
        yield prefix or 'value', value
    elif isinstance(value, dict):
        for key in sorted(value):
            child = f'{prefix}.{key}' if prefix else str(key)
            yield from _scalar_leaves(value[key], child)
    elif isinstance(value, list):
        for index, child_value in enumerate(value):
            yield from _scalar_leaves(child_value, f'{prefix}[{index}]')


def _artifact_text(item: dict[str, Any]) -> str:
    payload = item['json']
    hints = [item['path']]
    if isinstance(payload, dict):
        for key in ('phase', 'experiment', 'split', 'backend', 'mode', 'kind'):
            if key in payload:
                hints.append(str(payload[key]))
    return ' '.join(hints).lower()


def _summary_artifact(item: dict[str, Any]) -> bool:
    name = item['path'].lower()
    return any(token in name for token in (
        'summary', 'metric', 'manifest', 'audit', 'run.json', 'exit.json',
        'curve', 'comparison', 'report', 'complex', 'timing', 'memory',
    ))


SECTION_DEFINITIONS = (
    ('D1', ('d1', 'pool', 'diversity')),
    ('D2', ('d2', 'type-competition', 'competition', 'prefix')),
    ('F0/F1', ('fold', 'f0', 'f1', 'f-type')),
    ('Reward model', ('reward', 'rm-', 'score')),
    ('R1 / R0 / R01 / RB', ('r1', 'r0', 'r01', 'rb', 'selection')),
    ('Tau scan', ('tau', 'threshold')),
    ('Complex test', ('complex', 'eval_complex')),
    ('Correction diagnostics', ('diagnostic', 'corrected', 'recovered', 'added_fp')),
    ('Baselines and comparison', ('baseline', 't0', 'exceeds')),
    ('Timing and peak memory', ('timing', 'memory', 'gpu', 'wall', 'peak')),
)


def _section_items(artifacts: list[dict[str, Any]], keywords: tuple[str, ...]) -> list[dict[str, Any]]:
    return [item for item in artifacts
            if _summary_artifact(item) and any(keyword in _artifact_text(item) for keyword in keywords)]


def _render_values(items: list[dict[str, Any]]) -> list[str]:
    rows: list[tuple[str, str, Any]] = []
    for item in items:
        for field, value in _scalar_leaves(item['json']):
            rows.append((item['path'], field, value))
    if not rows:
        return ['No matching JSON summary artifact was produced.']
    lines = ['| Artifact | Field | Value |', '|---|---|---|']
    for artifact, field, value in rows:
        rendered = json.dumps(value, ensure_ascii=False) if not isinstance(value, str) else value
        lines.append(f'| `{artifact}` | `{field}` | {rendered} |')
    return lines


def _render_index(artifacts: list[dict[str, Any]]) -> list[str]:
    if not artifacts:
        return ['No JSON artifacts found.']
    lines = ['| Artifact | Bytes | SHA256 |', '|---|---:|---|']
    for item in artifacts:
        lines.append(f"| `{item['path']}` | {item['bytes']} | `{item['sha256']}` |")
    return lines


def render_report(run_dir: str | Path) -> str:
    artifacts = collect_artifacts(run_dir)
    if not artifacts:
        raise ValueError(f'no JSON artifacts under {run_dir}')
    lines = [
        '# Stage-3 Tree Refinement v1 Report',
        '',
        'This report is generated only from frozen JSON artifacts under the run directory.',
        'Missing sections mean that the corresponding queue phase has not emitted a JSON summary.',
        '',
        '## Artifact index',
        '',
        *_render_index(artifacts),
    ]
    descriptions = {
        'D1': 'Required rows: dev/test, oracle recovery, and candidate diversity.',
        'D2': 'Required rows: dev/test, confusion-pair counts before/after, and prefix-token consistency.',
        'F0/F1': 'Required rows: fold model metrics and train-pool versus dev-pool distributions.',
        'Reward model': 'Required rows: three-head dev AUC/Spearman, including the RoBERTa ablation when present.',
        'R1 / R0 / R01 / RB': 'Required rows: official metrics, 4.4 diagnostics, D2/R1 comparison, and candidate oracle rows.',
        'Tau scan': 'Required rows: dev tau curve and the frozen test tau.',
        'Complex test': 'Required rows: eval_complex_quick_f1.py output for each final test configuration.',
        'Correction diagnostics': 'Required rows include type_corrected, type_corrupted, events_recovered, and events_added_fp.',
        'Baselines and comparison': 'Required rows: T0, D2, and the EXCEEDS reference (TI 75.3 / TC 63.7 / AI 45.0 / AC 43.2 / EC 48.3).',
        'Timing and peak memory': 'Required rows: GPU wall time and peak memory for each phase.',
    }
    for title, keywords in SECTION_DEFINITIONS:
        lines.extend(['', f'## {title}', '', descriptions[title], '',
                      *_render_values(_section_items(artifacts, keywords))])
    lines.extend([
        '', '## v1 omissions', '',
        '- R2 Tree Refiner is not implemented in v1.',
        '- Multi-seed runs are not included in v1.',
        '- Tau and reward weights are not learned in v1.',
        '- Complex-test values are reported only when an eval_complex_quick_f1.py JSON artifact is present.',
        '',
    ])
    return '\n'.join(lines)


def _write_text_atomically(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(prefix=f'.{path.name}.', suffix='.tmp', dir=path.parent)
    try:
        with os.fdopen(fd, 'w', encoding='utf-8') as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_name, path)
    finally:
        if os.path.exists(temporary_name):
            os.unlink(temporary_name)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    _write_text_atomically(args.output, render_report(args.run_dir))


if __name__ == '__main__':
    main()
