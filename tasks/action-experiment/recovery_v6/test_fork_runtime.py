"""CPU integration tests for immutable full-state precision transitions."""
import hashlib
import json
from pathlib import Path
import random
import sys

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from checkpoint_state import save_checkpoint, load_checkpoint
from fork_runtime import validate_fork_source, copy_committed_prefix


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def hashes(root):
    return {str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in root.rglob('*') if path.is_file()}


@pytest.fixture
def source(tmp_path):
    old = tmp_path / 'old'
    old.mkdir()
    config = dict(gradient_accumulation=4, reward_version='r0_v2', policy_update_mask_mode='field_v2')
    init = tmp_path / 'sft'
    write(old / 'configuration.json', config)
    write(old / 'provenance.json', dict(init_adapter=str(init), input_sha256={}))
    committed = b'{"group_index": 0}\n'
    (old / 'groups.jsonl').write_bytes(committed + b'{"partial": true}\n')
    (old / 'dev-history.jsonl').write_bytes(b'{"step": 1200}\n')
    write(old / 'failure.json', {'known': 'original probability guard failure'})
    write(old / 'planned-train-order.json', {'ids': ['a', 'b']})
    write(old / 'dev/step-1200/metrics.json', {'TC_f1': .634804})
    write(old / 'dev/step-1600/partial.json', {'incomplete': True})
    write(old / 'checkpoints/step-1200/policy/adapter_config.json', {'source': 'selected'})
    (old / 'checkpoints/step-1200/policy/adapter_model.safetensors').write_bytes(b'fixture adapter weights')
    model = torch.nn.Linear(2, 1)
    optimizer = torch.optim.AdamW(model.parameters())
    model(torch.ones(1, 2)).sum().backward()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    payload = dict(step=1490, next_group=5960, pending=0, best_step=1200,
        best_key=[.634804, .768], early_stopping=dict(last_step=1200, bad_count=0),
        log_offsets={'groups.jsonl': len(committed), 'dev-history.jsonl': len(b'{"step": 1200}\n')},
        reference_sha256='frozen', probability_guards={'current_vs_old': {'responses': 23840}})
    checkpoint = save_checkpoint(old / 'recovery-checkpoints', 'exception-step1490', model, optimizer, payload)
    return old, checkpoint, tmp_path / 'new', config, init, model, optimizer


def validated(source):
    old, checkpoint, output, config, init, *_ = source
    return validate_fork_source(old, checkpoint, output,
        {**config, 'arithmetic': {'protocol': 'fp32-body-bf16-storage-v1'}}, init, lambda locks: None)


def test_fork_copies_exact_committed_prefix_and_preserves_source(source):
    old, checkpoint, output, *_ = source
    before = hashes(old)
    fork = validated(source)
    output.mkdir()
    transition = copy_committed_prefix(fork, output)
    assert (output / 'groups.jsonl').read_bytes() == b'{"group_index": 0}\n'
    assert (output / 'dev-history.jsonl').read_bytes() == (old / 'dev-history.jsonl').read_bytes()
    assert (output / 'dev/step-1200/metrics.json').is_file()
    assert not (output / 'dev/step-1600').exists()
    assert (output / 'best-adapter/adapter_config.json').is_file()
    assert not (output / 'failure.json').exists()
    assert not (output / 'recovery-checkpoints/latest.json').exists()
    assert transition['source_step'] == 1490 and transition['source_next_group'] == 5960
    assert transition['optimizer_restored'] and transition['rng_restored']
    assert transition['inherited_file_sha256']
    assert before == hashes(old)


def test_optimizer_parameters_rng_and_loop_state_restore_exactly(source):
    _, checkpoint, _, _, _, original, original_optimizer = source
    fork = validated(source)
    expected = torch.load(checkpoint / 'training-state.pt', weights_only=False)
    expected_torch = torch.rand(4, generator=torch.Generator().set_state(expected['rng']['torch_cpu']))
    python_rng = random.Random()
    python_rng.setstate(expected['rng']['python'])
    expected_python = python_rng.random()
    model = torch.nn.Linear(2, 1)
    optimizer = torch.optim.AdamW(model.parameters())
    payload = load_checkpoint(checkpoint, model, optimizer)
    assert payload == fork['payload']
    for actual, saved in zip(model.parameters(), original.parameters()):
        assert torch.equal(actual, saved)
    assert len(optimizer.state) == len(original_optimizer.state) == 2
    for state in optimizer.state.values():
        assert state['step'].item() == 1
    assert torch.equal(torch.rand(4), expected_torch)
    assert random.random() == expected_python


@pytest.mark.parametrize('change', ['config', 'corrupt', 'partial', 'offset', 'nested', 'existing', 'foreign'])
def test_rejects_unsafe_forks_before_creating_output(source, change):
    old, checkpoint, output, config, init, *_ = source
    if change == 'config':
        config['policy_update_mask_mode'] = 'off'
    elif change == 'corrupt':
        (checkpoint / 'adapter/model.pt').write_bytes(b'corrupt')
    elif change in {'partial', 'offset'}:
        state = torch.load(checkpoint / 'training-state.pt', weights_only=False)
        if change == 'partial':
            state['payload']['pending'] = 1
        else:
            state['payload']['log_offsets']['groups.jsonl'] = 3
        torch.save(state, checkpoint / 'training-state.pt')
        manifest = json.loads((checkpoint / 'manifest.json').read_text())
        manifest['files']['training-state.pt'] = hashlib.sha256((checkpoint / 'training-state.pt').read_bytes()).hexdigest()
        write(checkpoint / 'manifest.json', manifest)
    elif change == 'nested':
        output = old / 'new'
    elif change == 'existing':
        output.mkdir()
    elif change == 'foreign':
        checkpoint = output / 'foreign'
    with pytest.raises(ValueError):
        validate_fork_source(old, checkpoint, output, {**config, 'arithmetic': {'protocol': 'stable'}},
            init, lambda locks: None)
    if change != 'existing':
        assert not output.exists()


def test_source_locks_checked_before_fork(source):
    old, checkpoint, output, config, init, *_ = source
    def reject(_):
        raise ValueError('source lock changed')
    with pytest.raises(ValueError, match='source lock changed'):
        validate_fork_source(old, checkpoint, output, {**config, 'arithmetic': {}}, init, reject)
    assert not output.exists()


def test_train_order_must_match_source_even_when_lengths_match(source):
    from fork_runtime import validate_fork_train_order
    fork = validated(source)
    validate_fork_train_order(fork, {'ids': ['a', 'b']})
    with pytest.raises(ValueError, match='train order'):
        validate_fork_train_order(fork, {'ids': ['b', 'a']})


@pytest.mark.parametrize('batch', [True, 0, 3, 8])
def test_fork_rejects_unvalidated_evaluation_batch(source, batch):
    old, checkpoint, output, config, init, *_ = source
    with pytest.raises(ValueError, match='evaluation batch'):
        validate_fork_source(old, checkpoint, output,
            {**config, 'arithmetic': {}, 'evaluation_batch_size': batch}, init, lambda _: None)


def test_fork_records_explicit_evaluation_memory_transition(source):
    transition = validated(source)['transition']
    assert transition['previous_evaluation_batch_size'] == 4
    assert transition['evaluation_batch_size'] == 1


@pytest.mark.parametrize('batch', [2, 4])
def test_fork_records_verified_larger_evaluation_batch(source, batch):
    old, checkpoint, output, config, init, *_ = source
    fork = validate_fork_source(old, checkpoint, output,
        {**config, 'arithmetic': {}, 'evaluation_batch_size': batch}, init, lambda _: None)
    assert fork['transition']['evaluation_batch_size'] == batch


@pytest.mark.parametrize('change', ['smoke', 'init_adapter', 'train_sha256', 'temperature',
                                  'probability_guard', 'arithmetic', 'foreign_checkpoint'])
def test_resume_rejects_immutable_changes_without_writes(source, change):
    from fork_runtime import validate_resume_source
    old, checkpoint, _, config, init, *_ = source
    config = {**config, 'smoke': False, 'train_sha256': 'original', 'temperature': 1.,
              'probability_guard': {'hard_max_difference': 1.0}, 'arithmetic': {'protocol': 'stable'}}
    write(old / 'configuration.json', config)
    requested = dict(config)
    if change == 'init_adapter':
        init = init.with_name('wrong-sft')
    elif change == 'foreign_checkpoint':
        checkpoint = old.parent / 'other-run/recovery-checkpoints/same-shape'
    else:
        requested[change] = 'changed'
    before = hashes(old)
    with pytest.raises(ValueError):
        validate_resume_source(old, checkpoint, requested, init, lambda _: None)
    assert hashes(old) == before


def test_resume_accepts_source_metadata_but_rejects_plan_mutation(source):
    from fork_runtime import validate_resume_source, validate_resume_train_order
    old, checkpoint, _, config, init, *_ = source
    # Metadata changes alone do not alter the training domain.
    write(old / 'configuration.json', {**config, 'source_sha256': {'old': 'x'},
        'continuation_mode': 'full_state_precision_transition', 'precision_transition': {'old': 'source'}})
    before = hashes(old)
    validate_resume_source(old, checkpoint, {**config, 'source_sha256': {'new': 'y'}}, init, lambda _: None)
    validate_resume_train_order(old, {'ids': ['a', 'b']})
    with pytest.raises(ValueError, match='train order'):
        validate_resume_train_order(old, {'ids': ['b', 'a']})
    assert hashes(old) == before


def test_resume_rejects_checkpoint_arithmetic_before_run_writes(source):
    from fork_runtime import validate_resume_source
    old, checkpoint, _, config, init, *_ = source
    configured = {**config, 'arithmetic': {'protocol': 'stable'}}
    write(old / 'configuration.json', configured)
    before = hashes(old)
    with pytest.raises(ValueError, match='checkpoint arithmetic'):
        validate_resume_source(old, checkpoint, configured, init, lambda _: None)
    assert hashes(old) == before


def test_fork_rejects_missing_selected_adapter_before_new_run(source):
    old, _, output, *_ = source
    (old / 'checkpoints/step-1200/policy/adapter_model.safetensors').unlink()
    with pytest.raises(ValueError, match='selected adapter'):
        validated(source)
    assert not output.exists()


def test_copy_verifies_target_against_source_snapshot(source, monkeypatch):
    import fork_runtime
    _, _, output, *_ = source
    fork = validated(source)
    output.mkdir()
    original = fork_runtime.shutil.copytree
    def corrupt(src, dst, *args, **kwargs):
        result = original(src, dst, *args, **kwargs)
        target = Path(dst) / 'metrics.json'
        if target.exists():
            target.write_text('corrupted during copy')
        return result
    monkeypatch.setattr(fork_runtime.shutil, 'copytree', corrupt)
    with pytest.raises(ValueError, match='copy hash'):
        copy_committed_prefix(fork, output)


def test_copy_rejects_source_change_after_snapshot(source):
    old, _, output, *_ = source
    fork = validated(source)
    (old / 'planned-train-order.json').write_text('{"ids": ["changed"]}')
    output.mkdir()
    with pytest.raises(ValueError, match='copy hash'):
        copy_committed_prefix(fork, output)
