import json


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
