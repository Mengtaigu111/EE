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


def current_project_gpu_processes(project_root: str | Path) -> list[str]:
    """Return command lines for project processes currently using the GPU."""
    project_root = str(Path(project_root).resolve())
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader,nounits'],
            capture_output=True, text=True, check=True)
    except (FileNotFoundError, subprocess.CalledProcessError):
        return []
    commands = []
    for value in result.stdout.splitlines():
        value = value.strip()
        if not value.isdigit():
            continue
        ps = subprocess.run(['ps', '-p', value, '-o', 'args='], capture_output=True, text=True)
        command = ps.stdout.strip()
        if project_root in command:
            commands.append(command)
    return commands


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


def run_d1_phase(phase: str, run_dir: Path, model: str, adapter: Path,
                 data: Path, greedy: Path) -> None:
    require_tmux()
    require_fixed_python()
    project_root = Path(__file__).resolve().parents[3]
    processes = current_project_gpu_processes(project_root)
    if not c_gate(project_root / C_RUN_JSON, processes):
        raise RuntimeError('Experiment C must be completed and GPU idle before D1')
    run_dir.mkdir(parents=True, exist_ok=False)
    state = start_run(run_dir, phase, {'model': model, 'adapter': str(adapter),
                                      'data': str(data), 'greedy': str(greedy)})
    output = run_dir / f'{phase}.pool.jsonl'
    command = [FIXED_PYTHON, str(Path(__file__).with_name('sample_pool.py')),
               '--data', str(data), '--greedy', str(greedy), '--output', str(output),
               '--model', model, '--adapter', str(adapter)]
    try:
        run_stage(command, run_dir / f'{phase}.log', run_dir / f'{phase}.exit.json')
    except Exception as exc:
        state.update({'status': 'failed', 'error': repr(exc)})
        atomic_write_json(run_dir / f'{phase}.run.json', state)
        raise
    state.update({'status': 'completed', 'finished_at': datetime.now(timezone.utc).isoformat(),
                  'output': str(output)})
    atomic_write_json(run_dir / f'{phase}.run.json', state)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase', choices=('gate', 'd1-dev', 'd1-test'), default='gate')
    parser.add_argument('--c-run-json', type=Path, default=C_RUN_JSON)
    parser.add_argument('--run-dir', type=Path)
    parser.add_argument('--model', default='/opt/models/Qwen3-8B')
    parser.add_argument('--adapter', type=Path,
                        default=Path('tasks/stage2-event-tree/runs/T0-42-20260917-144424/training/adapter'))
    args = parser.parse_args()
    require_tmux()
    require_fixed_python()
    if args.phase == 'gate':
        project_root = Path(__file__).resolve().parents[3]
        processes = current_project_gpu_processes(project_root)
        if not c_gate(args.c_run_json, processes):
            raise RuntimeError('Experiment C is not complete or GPU process list was not empty')
        print(json.dumps({'status': 'ready', 'c_run_json': str(args.c_run_json),
                          'gpu_processes': processes}))
        return
    if args.run_dir is None:
        parser.error('--run-dir is required for D1 phases')
    split = 'dev' if args.phase == 'd1-dev' else 'test'
    root = Path(__file__).resolve().parents[1]
    data = root.parent / 'stage2-event-tree' / 'data' / 'none' / f'{split}.jsonl'
    greedy = root.parent / 'stage2-event-tree' / 'runs' / 'T0-42-20260917-144424' / f'{split}.predictions.jsonl'
    run_d1_phase(args.phase, args.run_dir, args.model, args.adapter, data, greedy)


if __name__ == '__main__':
    main()
