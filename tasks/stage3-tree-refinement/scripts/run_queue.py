"""Guarded, artifact-oriented Stage-3 phase queue primitives."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time
from typing import Any, Iterable

from common import atomic_write_json, sha256_file


FIXED_PYTHON = '/home/myluo/miniconda3/envs/qwen3meld-run/bin/python'
C_RUN_JSON = Path('tasks/stage2-event-tree/runs/C-42-20260918/run.json')


def require_tmux() -> None:
    if 'TMUX' not in os.environ:
        raise RuntimeError('GPU execution must run in a named tmux session')


def c_gate(run_json: str | Path, project_gpu_processes: Iterable[str]) -> bool:
    path = Path(run_json)
    if not path.exists():
        return False
    try:
        state = json.loads(path.read_text(encoding='utf-8'))
    except json.JSONDecodeError:
        return False
    return state.get('status') == 'completed' and not list(project_gpu_processes)


def assert_test_once(run_dir: str | Path) -> None:
    marker = Path(run_dir) / 'test.completed.json'
    if marker.exists():
        raise RuntimeError(f'test configurations may run only once: {marker}')


def require_fixed_python() -> None:
    if Path(sys.executable).resolve() != Path(FIXED_PYTHON).resolve():
        raise RuntimeError(f'use fixed interpreter: {FIXED_PYTHON}')


def start_run(run_dir: str | Path, phase: str, config: dict[str, Any]) -> dict[str, Any]:
    run_dir = Path(run_dir)
    state = {'phase': phase, 'status': 'running',
             'started_at': datetime.now(timezone.utc).isoformat(),
             'config': config, 'attempt': int(time.time() * 1000)}
    atomic_write_json(run_dir / f'{phase}.run.json', state)
    return state


def run_stage(command: list[str], log_path: str | Path, exit_path: str | Path) -> int:
    log_path = Path(log_path)
    exit_path = Path(exit_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.time()
    with log_path.open('x', encoding='utf-8') as stream:
        stream.write(shlex.join(command) + '\n')
        stream.flush()
        result = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT,
                                cwd=Path(__file__).resolve().parents[3])
    atomic_write_json(exit_path, {
        'returncode': result.returncode, 'wall_seconds': time.time() - started,
        'command': command, 'status': 'completed' if result.returncode == 0 else 'failed',
    })
    if result.returncode:
        raise subprocess.CalledProcessError(result.returncode, command)
    return result.returncode


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase', choices=('gate',), default='gate')
    parser.add_argument('--c-run-json', type=Path, default=C_RUN_JSON)
    args = parser.parse_args()
    require_tmux()
    require_fixed_python()
    if args.phase == 'gate':
        if not c_gate(args.c_run_json, []):
            raise RuntimeError('Experiment C is not complete or GPU process list was not empty')
        print(json.dumps({'status': 'ready', 'c_run_json': str(args.c_run_json)}))


if __name__ == '__main__':
    main()
