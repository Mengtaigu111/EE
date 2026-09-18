"""Render a Stage-3 report strictly from frozen run JSON artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def collect_json(run_dir: str | Path) -> dict[str, Any]:
    root = Path(run_dir)
    artifacts = {}
    for path in sorted(root.rglob('*.json')):
        try:
            artifacts[str(path.relative_to(root))] = json.loads(path.read_text(encoding='utf-8'))
        except json.JSONDecodeError as exc:
            raise ValueError(f'invalid JSON artifact: {path}') from exc
    return artifacts


def render_report(run_dir: str | Path) -> str:
    artifacts = collect_json(run_dir)
    if not artifacts:
        raise ValueError(f'no JSON artifacts under {run_dir}')
    lines = [
        '# Stage-3 Tree Refinement v1 Report',
        '',
        'This report is generated only from JSON artifacts under the run directory.',
        '',
        '## Artifact index',
        '',
    ]
    for path in artifacts:
        lines.append(f'- `{path}`')
    lines.extend(['', '## JSON summary', '', '```json',
                  json.dumps(artifacts, ensure_ascii=False, indent=2, sort_keys=True),
                  '```', '',
                  '## Scope omissions', '',
                  '- R2 Tree Refiner is not implemented in v1.',
                  '- Multi-seed runs are not included in v1.',
                  '- Tau and reward weights are not learned in v1.', ''])
    return '\n'.join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    output = args.output
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(render_report(args.run_dir), encoding='utf-8')


if __name__ == '__main__':
    main()
