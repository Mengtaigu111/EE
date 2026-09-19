"""Guarded, resumable, artifact-oriented Stage-3 phase queue primitives.

The queue deliberately keeps state in one small ``run.json`` per run
directory.  A phase can therefore be retried after a failed attempt without
mistaking an old process for the current one.  GPU entry points still require
both the fixed environment and a named tmux session; this module itself never
starts a GPU job unless a caller explicitly supplies a command to run.
"""

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
from typing import Any, Iterable, Sequence

from common import atomic_write_json, sha256_file


FIXED_PYTHON = '/home/myluo/miniconda3/envs/qwen3meld-run/bin/python'
C_RUN_JSON = Path('tasks/stage2-event-tree/runs/C-42-20260918/run.json')
# Phase A is the zero-training D1/D2 diagnosis. Only these phases wait for
# Stage-2 Experiment C and an idle GPU; Phase B F0/F1 may start independently.
C_GATE_PHASES = frozenset({'d1-dev', 'd1-test', 'd2-dev', 'd2-test'})
TERMINAL_STATES = frozenset({'completed', 'failed', 'blocked'})
PHASE_DEPENDENCIES = {
    'd1-dev': (),
    'd1-test': ('d1-dev',),
    'd2-dev': ('d1-dev',),
    'd2-test': ('d2-dev',),
    'f0': (),
    'f1': (),
    'f-type': ('f0', 'f1'),
    'rm-qwen': ('f0', 'f1', 'f-type'),
    'r1': ('rm-qwen', 'd2-test'),
    'r0': ('rm-qwen', 'd1-test'),
    'r01': ('r0', 'r1'),
    'rb': ('f0', 'f1'),
}


def phase_dependencies(phase: str) -> tuple[str, ...]:
    """Return same-run phase dependencies for the generic queue command."""
    return tuple(PHASE_DEPENDENCIES.get(phase, ()))


def phase_requires_c_gate(phase: str) -> bool:
    return phase in C_GATE_PHASES


class StaleAttemptError(RuntimeError):
    """Raised when an old worker tries to publish a newer run's result."""


def run_state_path(run_dir: str | Path) -> Path:
    return Path(run_dir) / 'run.json'


def phase_state_path(run_dir: str | Path, phase: str) -> Path:
    return Path(run_dir) / f'{phase}.run.json'


def load_run_state(run_dir: str | Path) -> dict[str, Any] | None:
    path = run_state_path(run_dir)
    if not path.exists():
        return None
    try:
        state = json.loads(path.read_text(encoding='utf-8'))
    except json.JSONDecodeError as exc:
        raise ValueError(f'invalid run state: {path}') from exc
    if not isinstance(state, dict):
        raise ValueError(f'run state must be an object: {path}')
    return state


def _new_attempt(previous: dict[str, Any] | None) -> int:
    old = int(previous.get('attempt', 0)) if previous else 0
    return max(time.time_ns(), old + 1)


def _snapshot(state: dict[str, Any]) -> dict[str, Any]:
    return {key: state[key] for key in
            ('phase', 'status', 'attempt', 'started_at', 'finished_at', 'error')
            if key in state}


def start_run(run_dir: str | Path, phase: str, config: dict[str, Any],
              *, dependencies: Sequence[str] = ()) -> dict[str, Any]:
    """Start a phase, resume a failed phase, or return a completed phase.

    A completed state is immutable and is returned unchanged.  A failed state
    gets a fresh attempt while retaining its failure in ``attempt_history``.
    A running state is never silently replaced: the caller must first resolve
    the live/stale worker explicitly.
    """
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    root_previous = load_run_state(run_dir)
    previous = root_previous
    if root_previous is not None and root_previous.get('phase') != phase:
        if root_previous.get('status') != 'completed':
            raise RuntimeError(
                f'run directory has active/incomplete phase {root_previous.get("phase")!r}')
        phase_path = phase_state_path(run_dir, phase)
        previous = None
        if phase_path.exists():
            try:
                previous = json.loads(phase_path.read_text(encoding='utf-8'))
            except json.JSONDecodeError as exc:
                raise ValueError(f'invalid phase state: {phase_path}') from exc
        else:
            # Preserve the completed phase before the root state is reused by
            # the next phase. This makes dependency checks resumable in-place.
            atomic_write_json(phase_state_path(run_dir, root_previous['phase']), root_previous)
    if previous is not None:
        if previous.get('phase') != phase:
            raise RuntimeError(
                f'run directory belongs to phase {previous.get("phase")!r}, not {phase!r}')
        if previous.get('config') != config:
            raise RuntimeError('cannot resume a run with a different configuration')
        if previous.get('status') == 'completed':
            return previous
        if previous.get('status') == 'running':
            raise RuntimeError(f'phase already running: {run_state_path(run_dir)}')

    attempt = _new_attempt(previous)
    history = list(previous.get('attempt_history', [])) if previous else []
    if previous is not None:
        history.append(_snapshot(previous))
    state = {
        'phase': phase,
        'status': 'running',
        'started_at': datetime.now(timezone.utc).isoformat(),
        'config': config,
        'dependencies': list(dependencies),
        'attempt': attempt,
        'attempt_history': history,
    }
    atomic_write_json(run_state_path(run_dir), state)
    return state


def transition_run(run_dir: str | Path, attempt: int, status: str,
                   **fields: Any) -> dict[str, Any]:
    """Atomically publish a terminal state for the current attempt."""
    if status not in TERMINAL_STATES:
        raise ValueError(f'invalid terminal status: {status}')
    state = load_run_state(run_dir)
    if state is None:
        raise RuntimeError(f'missing run state: {run_state_path(run_dir)}')
    if state.get('attempt') != attempt:
        raise StaleAttemptError(
            f'stale attempt {attempt}; current attempt is {state.get("attempt")}')
    if state.get('status') != 'running':
        raise RuntimeError(f'cannot transition from {state.get("status")!r}')
    state.update(fields)
    state['status'] = status
    state['finished_at'] = datetime.now(timezone.utc).isoformat()
    atomic_write_json(run_state_path(run_dir), state)
    atomic_write_json(phase_state_path(run_dir, state['phase']), state)
    return state


def artifact_provenance(paths: Iterable[str | Path], root: str | Path | None = None) -> list[dict[str, Any]]:
    """Return stable existence/size/hash records for input artifacts."""
    root_path = Path(root).resolve() if root is not None else None
    result = []
    for value in paths:
        path = Path(value)
        display = path.name
        if root_path is not None:
            try:
                display = str(path.resolve().relative_to(root_path))
            except ValueError:
                display = str(path)
        item: dict[str, Any] = {'path': display, 'exists': path.is_file()}
        if path.is_file():
            item.update({'size': path.stat().st_size, 'sha256': sha256_file(path)})
        else:
            item.update({'size': 0, 'sha256': None})
        result.append(item)
    return result


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
    except FileNotFoundError as exc:
        raise RuntimeError('nvidia-smi is required for the GPU-idle gate') from exc
    except subprocess.CalledProcessError as exc:
        raise RuntimeError('nvidia-smi failed; refusing to assume the GPU is idle') from exc
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
    root = Path(run_dir)
    markers = (root / 'test.completed.json', root / 'test.run.json')
    for marker in markers:
        if not marker.exists():
            continue
        if marker.name == 'test.completed.json':
            raise RuntimeError(f'test configurations may run only once: {marker}')
        try:
            state = json.loads(marker.read_text(encoding='utf-8'))
        except json.JSONDecodeError as exc:
            raise RuntimeError(f'invalid test marker: {marker}') from exc
        if state.get('status') == 'completed':
            raise RuntimeError(f'test configurations may run only once: {marker}')


def require_fixed_python() -> None:
    if Path(sys.executable).resolve() != Path(FIXED_PYTHON).resolve():
        raise RuntimeError(f'use fixed interpreter: {FIXED_PYTHON}')


def run_stage(command: list[str], log_path: str | Path, exit_path: str | Path) -> int:
    log_path = Path(log_path)
    exit_path = Path(exit_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.time()
    if exit_path.exists():
        try:
            previous = json.loads(exit_path.read_text(encoding='utf-8'))
        except json.JSONDecodeError as exc:
            raise RuntimeError(f'invalid stage exit artifact: {exit_path}') from exc
        if previous.get('status') == 'completed' and previous.get('returncode') == 0:
            return 0
    with log_path.open('a', encoding='utf-8') as stream:
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


def _dependency_state(root: Path, dependency: str) -> dict[str, Any] | None:
    dependency_state = root / f'{dependency}.run.json'
    if dependency_state.exists():
        try:
            return json.loads(dependency_state.read_text(encoding='utf-8'))
        except json.JSONDecodeError:
            return None
    current = load_run_state(root)
    if current and current.get('phase') == dependency:
        return current
    return None


def phase_gate(run_dir: str | Path, phase: str,
               dependencies: Sequence[str] | None = None,
               dependency_dirs: Sequence[str | Path] = ()) -> bool:
    """Check that every dependency has a completed state.

    Dependencies are looked up in the run directory first, then in each of
    ``dependency_dirs`` (other phases' run directories), in order.
    """
    roots = [Path(run_dir), *(Path(item) for item in dependency_dirs)]
    dependencies = phase_dependencies(phase) if dependencies is None else tuple(dependencies)
    for dependency in dependencies:
        state = next((found for found in (_dependency_state(root, dependency) for root in roots)
                      if isinstance(found, dict)), None)
        if not isinstance(state, dict) or state.get('status') != 'completed':
            return False
    return True


def _mark_test_completed(run_dir: str | Path, state: dict[str, Any]) -> None:
    atomic_write_json(Path(run_dir) / 'test.completed.json', {
        'phase': state['phase'], 'attempt': state['attempt'],
        'finished_at': state['finished_at'], 'status': 'completed',
    })


def run_resumable_phase(run_dir: str | Path, phase: str, command: list[str],
                        config: dict[str, Any], *, dependencies: Sequence[str] | None = None,
                        gpu: bool = False, c_run_json: str | Path = C_RUN_JSON,
                        project_root: str | Path | None = None,
                        dependency_dirs: Sequence[str | Path] = ()) -> dict[str, Any]:
    """Run one command with gates and resumable state artifacts."""
    if phase.endswith('-test'):
        assert_test_once(run_dir)
    if gpu:
        require_tmux()
        require_fixed_python()
        if not command or Path(command[0]).resolve() != Path(FIXED_PYTHON).resolve():
            raise RuntimeError(f'GPU phase command must start with fixed interpreter: {FIXED_PYTHON}')
        if phase_requires_c_gate(phase):
            root = Path(project_root) if project_root is not None else Path(__file__).resolve().parents[3]
            processes = current_project_gpu_processes(root)
            if not c_gate(c_run_json, processes):
                raise RuntimeError('Experiment C must be completed and GPU idle before D1/D2 Phase A')
    phase_dependencies = tuple(PHASE_DEPENDENCIES.get(phase, ())) if dependencies is None else tuple(dependencies)
    if not phase_gate(run_dir, phase, phase_dependencies, dependency_dirs):
        raise RuntimeError(f'phase dependencies are incomplete for {phase}: {list(phase_dependencies)}')
    state = start_run(run_dir, phase, config, dependencies=phase_dependencies)
    if state.get('status') == 'completed':
        if phase.endswith('-test') and not (Path(run_dir) / 'test.completed.json').exists():
            _mark_test_completed(run_dir, state)
        return state
    attempt = state['attempt']
    root = Path(run_dir)
    try:
        run_stage(command, root / f'{phase}.log', root / f'{phase}.exit.json')
    except Exception as exc:
        transition_run(root, attempt, 'failed', error=repr(exc))
        raise
    state = transition_run(root, attempt, 'completed', command=command,
                           checkpoint='dev-stop' if phase.endswith('-dev') else None)
    if phase.endswith('-test'):
        _mark_test_completed(root, state)
    return state


def run_d1_phase(phase: str, run_dir: Path, model: str, adapter: Path,
                 data: Path, greedy: Path) -> None:
    require_tmux()
    require_fixed_python()
    if phase.endswith('-test'):
        assert_test_once(run_dir)
    project_root = Path(__file__).resolve().parents[3]
    processes = current_project_gpu_processes(project_root)
    if not c_gate(project_root / C_RUN_JSON, processes):
        raise RuntimeError('Experiment C must be completed and GPU idle before D1')
    config = {'model': model, 'adapter': str(adapter), 'data': str(data), 'greedy': str(greedy)}
    state = start_run(run_dir, phase, config)
    if state.get('status') == 'completed':
        return
    attempt = state['attempt']
    output = run_dir / f'{phase}.pool.jsonl'
    command = [FIXED_PYTHON, str(Path(__file__).with_name('sample_pool.py')),
               '--data', str(data), '--greedy', str(greedy), '--output', str(output),
               '--model', model, '--adapter', str(adapter)]
    try:
        run_stage(command, run_dir / f'{phase}.log', run_dir / f'{phase}.exit.json')
    except Exception as exc:
        transition_run(run_dir, attempt, 'failed', error=repr(exc))
        raise
    state = transition_run(run_dir, attempt, 'completed', output=str(output),
                           provenance=artifact_provenance([data, greedy, adapter], root=run_dir))
    if phase.endswith('-test'):
        _mark_test_completed(run_dir, state)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase', choices=('gate', 'd1-dev', 'd1-test', *PHASE_DEPENDENCIES), default='gate')
    parser.add_argument('--c-run-json', type=Path, default=C_RUN_JSON)
    parser.add_argument('--run-dir', type=Path)
    parser.add_argument('--command', nargs=argparse.REMAINDER,
                        help='command for a generic phase; use --gpu for GPU work')
    parser.add_argument('--gpu', action='store_true',
                        help='enforce the tmux, fixed-python, C-complete, GPU-idle gates')
    parser.add_argument('--dependency-dir', type=Path, action='append', default=[],
                        help='run directory of a completed dependency phase (repeatable)')
    parser.add_argument('--model', default='/opt/models/Qwen3-8B')
    parser.add_argument('--adapter', type=Path,
                        default=Path('tasks/stage2-event-tree/runs/T0-42-20260917-144424/training/adapter'))
    args = parser.parse_args()
    if args.phase == 'gate':
        project_root = Path(__file__).resolve().parents[3]
        processes = current_project_gpu_processes(project_root)
        if not c_gate(args.c_run_json, processes):
            raise RuntimeError('Experiment C is not complete or GPU process list was not empty')
        print(json.dumps({'status': 'ready', 'c_run_json': str(args.c_run_json),
                          'gpu_processes': processes}))
        return
    if args.run_dir is None:
        parser.error('--run-dir is required for phase execution')
    if args.phase in ('d1-dev', 'd1-test') and args.command is None:
        require_fixed_python()
        split = 'dev' if args.phase == 'd1-dev' else 'test'
        root = Path(__file__).resolve().parents[1]
        data = root.parent / 'stage2-event-tree' / 'data' / 'none' / f'{split}.jsonl'
        greedy = root.parent / 'stage2-event-tree' / 'runs' / 'T0-42-20260917-144424' / f'{split}.predictions.jsonl'
        run_d1_phase(args.phase, args.run_dir, args.model, args.adapter, data, greedy)
        return
    if not args.command:
        parser.error('--command is required for generic phases')
    state = run_resumable_phase(
        args.run_dir, args.phase, args.command, {'command': args.command},
        gpu=args.gpu, c_run_json=args.c_run_json, dependency_dirs=args.dependency_dir,
    )
    print(json.dumps(state, ensure_ascii=False))


if __name__ == '__main__':
    main()
