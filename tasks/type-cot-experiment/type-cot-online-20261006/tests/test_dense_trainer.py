"""Exercise the explicit schedule in the actual Transformers callback chain."""
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
from transformers import TrainerControl, TrainerState
from transformers.trainer_callback import CallbackHandler, DefaultFlowCallback

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1] / 'stage2-event-tree/curriculum'))


def dense_module():
    import importlib.util
    assert importlib.util.find_spec('dense_trainer') is not None, 'Explicit schedule implementation is missing'
    import dense_trainer
    return dense_trainer


def handler(steps):
    callback = dense_module().ExplicitEvaluationCallback(steps)
    return CallbackHandler([DefaultFlowCallback(), callback], None, None, None, None)


def test_exact_formal_schedule_overrides_periodic_flags():
    module = dense_module()
    expected = [401, 802, 1203, 1403, 1604, 1804, 2005, 2205, 2406, 2606, 2807, 3007, 3208]
    assert module.EVAL_STEPS == expected
    callbacks = handler(expected)
    args = SimpleNamespace(logging_strategy='no', eval_strategy='steps', save_strategy='steps',
                           eval_delay=0, logging_first_step=False, max_steps=3208)
    state = TrainerState(max_steps=3208, eval_steps=401, save_steps=401, logging_steps=1)
    evaluated, saved = [], []
    for step in range(1, 3209):
        state.global_step = step
        control = callbacks.on_step_begin(args, state, TrainerControl())
        control = callbacks.on_step_end(args, state, control)
        if control.should_evaluate:
            evaluated.append(step)
        if control.should_save:
            saved.append(step)
    assert evaluated == expected
    assert saved == expected
    assert 1402 not in evaluated and 1403 in evaluated and 1604 in saved


def test_feedback_stop_preserves_checkpoint_and_excludes_later_steps():
    from dev_feedback import FeedbackController
    module = dense_module()
    callbacks = handler(module.EVAL_STEPS)
    args = SimpleNamespace(logging_strategy='no', eval_strategy='no', save_strategy='no',
                           logging_first_step=False, max_steps=3208)
    state = TrainerState(max_steps=3208)
    feedback = FeedbackController()
    evaluations, checkpoints = [], []
    for step in range(1, 3209):
        state.global_step = step
        control = callbacks.on_step_begin(args, state, TrainerControl())
        control = callbacks.on_step_end(args, state, control)
        if control.should_evaluate:
            evaluations.append(step)
            action = feedback.update(step, .5)
            control.should_training_stop |= action['stop']
        if control.should_save:
            checkpoints.append(step)
        if control.should_training_stop:
            break
    assert state.global_step == 2005
    assert evaluations == checkpoints == module.EVAL_STEPS[:7]
    assert 1604 in checkpoints


@pytest.mark.parametrize('steps', [[], [0], [2, 2], [4, 2], [True]])
def test_rejects_invalid_schedule(steps):
    with pytest.raises(ValueError):
        dense_module().ExplicitEvaluationCallback(steps)


def test_cpu_trainer_evaluates_only_list_and_retains_stop_checkpoint(tmp_path):
    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import GPT2Config, GPT2LMHeadModel, TrainingArguments, default_data_collator
    from sorted_first_sampler import IndexedDataset, ObservedCollator
    from online_dev_trainer import OFFICIAL_KEYS, validate_online_checkpoint
    torch.set_num_threads(1)
    model = get_peft_model(GPT2LMHeadModel(GPT2Config(vocab_size=8, n_layer=1,
                n_head=1, n_embd=8, n_positions=8)), LoraConfig(task_type='CAUSAL_LM',
                r=2, lora_alpha=4, target_modules=['c_attn']))
    rows = [{'input_ids': [1, 2, 3], 'labels': [-100, 2, 3]} for _ in range(16)]
    dataset = IndexedDataset(rows, [str(i) for i in range(16)])
    steps = [1, 2, 3, 5, 6, 7, 9, 10]
    evaluated = []
    def evaluate(model, step):
        evaluated.append(step)
        return {'official': {key: .5 for key in OFFICIAL_KEYS}}
    args = TrainingArguments(output_dir=str(tmp_path/'checkpoints'), use_cpu=True,
                per_device_train_batch_size=1, max_steps=10, num_train_epochs=1,
                learning_rate=.0001, save_strategy='no', eval_strategy='no',
                save_total_limit=None, report_to='none', remove_unused_columns=False,
                dataloader_num_workers=0, logging_steps=1, disable_tqdm=True)
    trainer = dense_module().DenseOnlineDevTrainer(model=model, args=args,
                train_dataset=dataset, eval_dataset=rows,
                data_collator=ObservedCollator(default_data_collator),
                observation_path=tmp_path/'observed-order.jsonl',
                dev_evaluator=evaluate, feedback_path=tmp_path/'feedback.jsonl',
                warmup_steps=1, checkpoint_steps=steps)
    import train_tree
    assert hasattr(train_tree, 'make_handoff_callback'), 'Freshness callback is missing'
    model.save_pretrained(tmp_path/'source-adapter')
    trainer.add_callback(train_tree.make_handoff_callback(trainer,
        {'source_adapter': str(tmp_path/'source-adapter')}, tmp_path/'handoff.json'))
    trainer.train()
    import json
    evidence = json.loads((tmp_path/'handoff.json').read_text())
    assert evidence['model_exact'] and evidence['optimizer_state_empty']
    assert evidence['scheduler_last_epoch'] == evidence['global_step'] == 0
    assert evidence['feedback']['eval_count'] == 0
    assert trainer.state.global_step == 9
    assert evaluated == steps[:7]
    checkpoints = sorted(int(p.name.split('-')[-1]) for p in (tmp_path/'checkpoints').glob('checkpoint-*'))
    assert checkpoints == steps[:7]
    for step in checkpoints:
        validate_online_checkpoint(tmp_path/'checkpoints'/f'checkpoint-{step}', step, 10)


def test_handoff_rejects_foreign_run_and_unfinished_cot(tmp_path):
    import importlib.util
    import json
    assert importlib.util.find_spec('train_tree') is not None, 'Tree handoff validation is missing'
    from train_tree import inspect_handoff_source
    cot = tmp_path/'run/cot'
    adapter = cot/'final-adapter'
    adapter.mkdir(parents=True)
    (adapter/'adapter_model.safetensors').write_bytes(b'weights')
    (adapter/'adapter_config.json').write_text('{}')
    (cot/'config.json').write_text(json.dumps({'initialization': 'fresh base + new LoRA'}))
    (cot/'train_metrics.json').write_text(json.dumps({'status': 'running'}))
    with pytest.raises(ValueError, match='completed'):
        inspect_handoff_source(adapter, tmp_path/'run/tree')
    from config import sha
    (cot/'train_metrics.json').write_text(json.dumps({'status': 'completed',
        'final_adapter_sha256': sha(adapter/'adapter_model.safetensors')}))
    evidence = inspect_handoff_source(adapter, tmp_path/'run/tree')
    assert len(evidence['source_sha256']) == 4
    (adapter/'adapter_model.safetensors').write_bytes(b'tampered')
    with pytest.raises(ValueError, match='hash'):
        inspect_handoff_source(adapter, tmp_path/'run/tree')
    (adapter/'adapter_model.safetensors').write_bytes(b'weights')
    with pytest.raises(ValueError, match='this run'):
        inspect_handoff_source(adapter, tmp_path/'foreign/tree')
    (cot/'config.json').write_text(json.dumps({'initialization': 'resumed'}))
    with pytest.raises(ValueError, match='fresh'):
        inspect_handoff_source(adapter, tmp_path/'run/tree')
