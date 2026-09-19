import json
import pytest


def test_c_gate_requires_completed_status_and_no_gpu_processes(tmp_path):
    from run_queue import c_gate

    state = tmp_path / 'run.json'
    state.write_text(json.dumps({'status': 'completed'}), encoding='utf-8')
    assert c_gate(state, []) is True
    assert c_gate(state, ['python predict.py']) is False
    state.write_text(json.dumps({'status': 'running'}), encoding='utf-8')
    assert c_gate(state, []) is False


def test_test_once_rejects_existing_completion_marker(tmp_path):
    from run_queue import assert_test_once

    assert_test_once(tmp_path)
    (tmp_path / 'test.completed.json').write_text('{}', encoding='utf-8')
    try:
        assert_test_once(tmp_path)
    except RuntimeError as exc:
        assert 'once' in str(exc)
    else:
        raise AssertionError('expected duplicate test guard')


def test_gpu_queue_requires_tmux_environment(monkeypatch):
    from run_queue import require_tmux

    monkeypatch.delenv('TMUX', raising=False)
    try:
        require_tmux()
    except RuntimeError as exc:
        assert 'tmux' in str(exc)
    else:
        raise AssertionError('expected tmux guard')


def test_run_state_transitions_are_atomic_and_preserve_attempt_history(tmp_path):
    from run_queue import StaleAttemptError, start_run, transition_run

    state = start_run(tmp_path, 'd1-dev', {'seed': 42})
    assert json.loads((tmp_path / 'run.json').read_text())['status'] == 'running'

    with pytest.raises(StaleAttemptError):
        transition_run(tmp_path, state['attempt'] - 1, 'completed')

    completed = transition_run(tmp_path, state['attempt'], 'completed', output='pool.jsonl')
    assert completed['status'] == 'completed'
    assert completed['output'] == 'pool.jsonl'
    assert not list(tmp_path.glob('*.tmp'))

    resumed = start_run(tmp_path, 'd1-dev', {'seed': 42})
    assert resumed['status'] == 'completed'
    assert resumed['attempt'] == state['attempt']


def test_failed_run_can_resume_with_new_attempt_and_stale_attempt_cannot_finish(tmp_path):
    from run_queue import StaleAttemptError, start_run, transition_run

    first = start_run(tmp_path, 'd2-dev', {})
    transition_run(tmp_path, first['attempt'], 'failed', error='boom')
    second = start_run(tmp_path, 'd2-dev', {})
    assert second['attempt'] != first['attempt']
    with pytest.raises(StaleAttemptError):
        transition_run(tmp_path, first['attempt'], 'completed')
    assert len(second['attempt_history']) == 1
    assert second['attempt_history'][0]['status'] == 'failed'


def test_phase_gate_requires_completed_dependencies(tmp_path):
    from run_queue import phase_gate, start_run, transition_run

    assert phase_gate(tmp_path, 'd2-dev', ['d1-dev']) is False
    d1 = start_run(tmp_path, 'd1-dev', {})
    transition_run(tmp_path, d1['attempt'], 'completed')
    assert phase_gate(tmp_path, 'd2-dev', ['d1-dev']) is True


def test_phase_dependencies_keep_folds_independent_of_phase_a_c_gate():
    from run_queue import phase_dependencies, phase_requires_c_gate

    assert phase_requires_c_gate('d1-dev') is True
    assert phase_requires_c_gate('d2-test') is True
    assert phase_requires_c_gate('f0') is False
    assert phase_requires_c_gate('f1') is False
    assert phase_dependencies('f0') == ()
    assert phase_dependencies('f1') == ()
    assert phase_dependencies('f-type') == ('f0', 'f1')


def test_gpu_f0_does_not_call_c_gate(monkeypatch, tmp_path):
    import run_queue

    monkeypatch.setattr(run_queue, 'require_tmux', lambda: None)
    monkeypatch.setattr(run_queue, 'require_fixed_python', lambda: None)
    monkeypatch.setattr(run_queue, 'c_gate', lambda *_: pytest.fail('F0 must not use C gate'))
    monkeypatch.setattr(run_queue, 'current_project_gpu_processes',
                        lambda *_: pytest.fail('F0 must not inspect C GPU gate'))
    state = run_queue.run_resumable_phase(
        tmp_path, 'f0', [run_queue.FIXED_PYTHON, '-c', 'pass'], {}, gpu=True)
    assert state['status'] == 'completed'


def test_multiple_phases_share_run_dir_without_losing_dependency_state(tmp_path):
    from run_queue import phase_gate, start_run, transition_run

    first = start_run(tmp_path, 'd1-dev', {})
    transition_run(tmp_path, first['attempt'], 'completed')
    second = start_run(tmp_path, 'd2-dev', {}, dependencies=['d1-dev'])
    assert second['phase'] == 'd2-dev'
    assert phase_gate(tmp_path, 'd2-dev', ['d1-dev']) is True
    assert json.loads((tmp_path / 'd1-dev.run.json').read_text())['status'] == 'completed'


def test_test_phase_publishes_a_single_completion_marker(tmp_path):
    from run_queue import run_resumable_phase

    state = run_resumable_phase(
        tmp_path, 'd1-test', ['/bin/sh', '-c', 'exit 0'], {}, gpu=False,
        dependencies=(),
    )
    assert state['status'] == 'completed'
    assert (tmp_path / 'test.completed.json').exists()


def test_provenance_records_missing_and_existing_files(tmp_path):
    from run_queue import artifact_provenance

    payload = tmp_path / 'input.json'
    payload.write_text('{"x": 1}\n', encoding='utf-8')
    result = artifact_provenance([payload, tmp_path / 'missing.json'])
    assert result[0]['path'] == 'input.json'
    assert len(result[0]['sha256']) == 64
    assert result[1]['exists'] is False


def test_phase_gate_accepts_dependencies_from_other_run_directories(tmp_path):
    from run_queue import phase_gate, start_run, transition_run

    other = tmp_path / 'D1'
    state = start_run(other, 'd1-dev', {})
    assert phase_gate(tmp_path / 'D2', 'd2-dev', ['d1-dev'], dependency_dirs=[other]) is False
    transition_run(other, state['attempt'], 'completed')
    assert phase_gate(tmp_path / 'D2', 'd2-dev', ['d1-dev'], dependency_dirs=[other]) is True
    assert phase_gate(tmp_path / 'D2', 'd2-dev', ['d1-dev']) is False
