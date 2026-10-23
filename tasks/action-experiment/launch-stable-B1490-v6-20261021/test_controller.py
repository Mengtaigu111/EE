import importlib.util
import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

SPEC = importlib.util.spec_from_file_location('stable_b_controller', Path(__file__).with_name('run_controller.py'))
controller = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(controller)


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def complete_output(output):
    point = output / 'recovery-checkpoints/complete-step1500-fixture'
    point.mkdir(parents=True)
    (point / 'training-state.pt').write_bytes(b'trusted fixture; no model loaded')
    write(point / 'manifest.json', {'format_version': 1,
        'files': {'training-state.pt': controller.digest(point / 'training-state.pt')}})
    write(point.parent / 'latest.json', {'checkpoint': point.name,
        'manifest_sha256': controller.digest(point / 'manifest.json')})
    best = output / 'best-adapter'
    best.mkdir()
    (best / 'adapter_model.safetensors').write_bytes(b'best weights')
    write(best / 'adapter_config.json', {'fixture': True})
    write(output / 'train_metrics.json', dict(status='completed', smoke=False,
        policy_update_mask_mode='field_v2', global_step=1500,
        arithmetic={'protocol': 'fp32-body-bf16-storage-v1'},
        best_adapter_sha256=controller.digest(best / 'adapter_model.safetensors'),
        best_adapter_config_sha256=controller.digest(best / 'adapter_config.json')))
    return point


def accepted_result():
    return {'status': 'completed', 'production_runs_modified': False, 'stages': {
        'real_forward': [dict(run=label, group=group, sample=0, max_difference=.0001, tokens=12)
                         for label in ('A', 'B') for group in range(3)],
        'optimizer': dict(samples=[dict(group=group, sample=sample, tokens=12, loss=.1, max_difference=.0001)
            for group in range(5960,5964) for sample in range(4)],
            gradient_norm=.5, initial_step=1490, final_step=1491, policy_changed=True, reference_unchanged=True),
        'checkpoint': dict(rng_equal=True, policy_equal=True, step=1491),
        'longest_train': dict(prompt_tokens=1200, completion_tokens=768, total_tokens=1968,
                              finite_gradients=True, peak_allocated_bytes=20_000_000_000),
        'longest_dev': dict(batch_size=4, prompt_tokens=1406, forced_generated_tokens=768,
                            peak_allocated_bytes=21_000_000_000),
        'dev_batch_benchmark': [dict(batch_size=4, status='passed', wall_seconds=1.,
                                    peak_allocated_bytes=21_000_000_000)]}}


def test_one_b_only_and_no_implicit_resume(tmp_path):
    launch, output = tmp_path / 'launch', tmp_path / 'newB'
    launch.mkdir()
    commands = []
    def start(command, log):
        commands.append(command)
        complete_output(output)
        return SimpleNamespace(pid=123, wait=lambda: 0)
    controller.run_controller(launch, output, verify=lambda _: 4, start=start)
    assert len(commands) == 1
    command = commands[0]
    assert '--fork-from-run' in command and str(controller.OLD_B) in command
    assert '--policy-update-mask' in command and 'field_v2' in command
    assert command[command.index('--evaluation-batch-size') + 1] == '4'
    assert not any('A800-recoverable' in value for value in command)
    assert controller.read(launch / 'pipeline-state.json')['status'] == 'completed'
    controller.run_controller(launch, output, verify=lambda _: 4, start=start)
    assert len(commands) == 1


def test_failed_stage_is_blocked_without_explicit_resume(tmp_path):
    launch, output = tmp_path / 'launch', tmp_path / 'newB'
    launch.mkdir()
    calls = []
    def fail(command, log):
        calls.append(command)
        return SimpleNamespace(pid=42, wait=lambda: 7)
    with pytest.raises(RuntimeError):
        controller.run_controller(launch, output, verify=lambda _: 4, start=fail)
    original = (launch / 'pipeline-state.json').read_bytes()
    with pytest.raises(ValueError, match='explicit'):
        controller.run_controller(launch, output, verify=lambda _: 4, start=fail)
    assert len(calls) == 1
    assert (launch / 'pipeline-state.json').read_bytes() == original


def test_resume_requires_checkpoint_in_new_run(tmp_path):
    launch, output = tmp_path / 'launch', tmp_path / 'newB'
    launch.mkdir()
    point = complete_output(tmp_path / 'otherB')
    with pytest.raises(ValueError, match='new B run'):
        controller.run_controller(launch, output, point, verify=lambda _: 4,
            start=lambda *_: pytest.fail('must not launch'))


def test_explicit_resume_uses_only_new_checkpoint(tmp_path):
    launch, output = tmp_path / 'launch', tmp_path / 'newB'
    launch.mkdir()
    point = complete_output(output)
    (output / 'train_metrics.json').unlink()
    seen = []
    def start(command, log):
        seen.append(command)
        return SimpleNamespace(pid=24, wait=lambda: 1)
    with pytest.raises(RuntimeError):
        controller.run_controller(launch, output, point, verify=lambda _: 4, start=start)
    assert '--fork-from-run' not in seen[0]
    assert str(point) in seen[0]


def test_zero_exit_without_completion_is_failure(tmp_path):
    launch, output = tmp_path / 'launch', tmp_path / 'newB'
    launch.mkdir()
    with pytest.raises(RuntimeError, match='completion'):
        controller.run_controller(launch, output, verify=lambda _: 4,
            start=lambda *_: SimpleNamespace(pid=1, wait=lambda: 0))
    assert controller.read(launch / 'pipeline-state.json')['status'] == 'failed'


def test_completed_metrics_cannot_hide_corrupt_checkpoint(tmp_path):
    output = tmp_path / 'newB'
    point = complete_output(output)
    (point / 'training-state.pt').write_bytes(b'corrupted after completion')
    assert not controller.completed(output)


def test_acceptance_requires_success_exit_passed_batch_and_unchanged_runtime(tmp_path):
    runtime = tmp_path / 'stable_arithmetic.py'
    runtime.write_text('numerical runtime')
    write(tmp_path / 'runtime-source-sha256.json', {str(runtime): controller.digest(runtime)})
    write(tmp_path / 'result.json', accepted_result())
    (tmp_path / 'gpu.exit').write_text('0\n')
    assert controller.verify_acceptance(tmp_path, required_runtime={str(runtime)}) == 4
    runtime.write_text('changed numerical runtime')
    with pytest.raises(ValueError, match='runtime'):
        controller.verify_acceptance(tmp_path, required_runtime={str(runtime)})


def test_acceptance_rejects_running_result_or_unpassed_batch(tmp_path):
    write(tmp_path / 'runtime-source-sha256.json', {})
    (tmp_path / 'gpu.exit').write_text('0\n')
    for status, benchmark in [('running', 'passed'), ('completed', 'oom')]:
        write(tmp_path / 'result.json', {'status': status, 'stages': {
            'longest_dev': {'batch_size': 4, 'forced_generated_tokens': 768},
            'dev_batch_benchmark': [{'batch_size': 4, 'status': benchmark}]}})
        with pytest.raises(ValueError):
            controller.verify_acceptance(tmp_path, required_runtime=set())


def test_controller_rejects_batch_override_that_differs_from_acceptance(tmp_path):
    launch = tmp_path / 'launch'
    launch.mkdir()
    with pytest.raises(ValueError, match='batch'):
        controller.run_controller(launch, tmp_path / 'newB', verify=lambda _: 4,
            evaluation_batch_size=2, start=lambda *_: pytest.fail('unverified batch launched'))


@pytest.mark.parametrize('missing', ['real_forward', 'optimizer', 'checkpoint', 'longest_train'])
def test_acceptance_rejects_missing_required_stage(tmp_path, missing):
    result = accepted_result()
    result['stages'].pop(missing)
    write(tmp_path / 'result.json', result)
    write(tmp_path / 'runtime-source-sha256.json', {})
    (tmp_path / 'gpu.exit').write_text('0\n')
    with pytest.raises(ValueError):
        controller.verify_acceptance(tmp_path, required_runtime=set())


@pytest.mark.parametrize('change', ['forward_count', 'forward_nan', 'forward_large',
    'sample_count', 'sample_duplicate', 'sample_bad_loss', 'gradient_nan', 'policy_unchanged',
    'reference_changed', 'rng_different', 'train_short', 'train_bad_total', 'train_bad_gradient', 'train_zero_peak'])
def test_acceptance_rejects_invalid_numerical_or_training_evidence(tmp_path, change):
    result = accepted_result()
    stages = result['stages']
    if change == 'forward_count': stages['real_forward'].pop()
    elif change == 'forward_nan': stages['real_forward'][0]['max_difference'] = float('nan')
    elif change == 'forward_large': stages['real_forward'][0]['max_difference'] = .001
    elif change == 'sample_count': stages['optimizer']['samples'].pop()
    elif change == 'sample_duplicate': stages['optimizer']['samples'][0] = copy.deepcopy(stages['optimizer']['samples'][1])
    elif change == 'sample_bad_loss': stages['optimizer']['samples'][0]['loss'] = float('inf')
    elif change == 'gradient_nan': stages['optimizer']['gradient_norm'] = float('nan')
    elif change == 'policy_unchanged': stages['optimizer']['policy_changed'] = False
    elif change == 'reference_changed': stages['optimizer']['reference_unchanged'] = False
    elif change == 'rng_different': stages['checkpoint']['rng_equal'] = False
    elif change == 'train_short': stages['longest_train']['completion_tokens'] = 64
    elif change == 'train_bad_total': stages['longest_train']['total_tokens'] = 100
    elif change == 'train_bad_gradient': stages['longest_train']['finite_gradients'] = False
    elif change == 'train_zero_peak': stages['longest_train']['peak_allocated_bytes'] = 0
    write(tmp_path / 'result.json', result)
    write(tmp_path / 'runtime-source-sha256.json', {})
    (tmp_path / 'gpu.exit').write_text('0\n')
    with pytest.raises(ValueError):
        controller.verify_acceptance(tmp_path, required_runtime=set())
