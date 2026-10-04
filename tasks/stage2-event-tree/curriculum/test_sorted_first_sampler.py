"""Regression tests exercise the real CPU Trainer/Accelerate epoch transition."""
import json
import sys
from pathlib import Path

import pytest
import torch
from torch.utils.data import RandomSampler

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
from sorted_first_sampler import (SortedFirstSampler, IndexedDataset, ObservedCollator,
                                  SortedFirstTrainer, sampling_audit)


def test_sampler_is_epoch_aware_and_not_random_sampler():
    sampler = SortedFirstSampler(32, seed=42)
    assert not isinstance(sampler, RandomSampler)
    assert list(sampler) == [(0, i) for i in range(32)]
    sampler.set_epoch(1)
    expected = torch.randperm(32, generator=torch.Generator().manual_seed(43)).tolist()
    assert list(sampler) == [(1, i) for i in expected]
    assert expected != list(range(32))
    assert len(set(expected)) == 32
    assert list(sampler) == list(sampler)
    sampler.set_epoch(2)
    assert [i for _, i in sampler] != expected


@pytest.mark.parametrize("size", [32, 28])
def test_real_cpu_trainer_keeps_sampler_and_complete_epochs(tmp_path, size):
    from peft import LoraConfig, get_peft_model
    from transformers import GPT2Config, GPT2LMHeadModel, TrainingArguments
    from curriculum_trainer import validate_resume_checkpoint
    from model_utils import AssistantCollator
    torch.set_num_threads(1)
    rows = [{'input_ids': [1, 2, 3], 'attention_mask': [1, 1, 1],
             'labels': [-100, 2, 3]} for _ in range(size)]
    model = get_peft_model(GPT2LMHeadModel(GPT2Config(vocab_size=8, n_positions=8,
        n_embd=8, n_layer=1, n_head=1)), LoraConfig(task_type='CAUSAL_LM',
        r=2, lora_alpha=4, target_modules=['c_attn']))
    collator = ObservedCollator(AssistantCollator(0))
    dataset = IndexedDataset(rows, [f'row-{i}' for i in range(size)])
    args = TrainingArguments(output_dir=str(tmp_path / 'checkpoints'), use_cpu=True,
        per_device_train_batch_size=1, gradient_accumulation_steps=16, max_steps=4,
        save_steps=2, save_strategy='steps', logging_steps=1, report_to='none',
        remove_unused_columns=False, label_names=['labels'], seed=42, data_seed=42,
        disable_tqdm=True, dataloader_num_workers=0)
    observed = tmp_path / 'observed-order.jsonl'
    trainer = SortedFirstTrainer(model=model, args=args, train_dataset=dataset,
        data_collator=collator, observation_path=observed, sampling_seed=42)
    loader = trainer.get_train_dataloader()
    assert isinstance(loader.batch_sampler.sampler, SortedFirstSampler)
    assert not isinstance(loader.batch_sampler.sampler, RandomSampler)
    trainer.train()
    records = [json.loads(line) for line in observed.read_text().splitlines()]
    assert len(records) == size * 2
    assert [r['index'] for r in records[:size]] == list(range(size))
    expected = torch.randperm(size, generator=torch.Generator().manual_seed(43)).tolist()
    assert [r['index'] for r in records[size:]] == expected
    assert [r['epoch'] for r in records] == [0] * size + [1] * size
    assert [sum(r['optimizer_step_before'] == step for r in records)
            for step in range(4)] == [16, size - 16, 16, size - 16]
    audit = sampling_audit(records, dataset.ids, seed=42, expected_epochs=2,
        global_step=trainer.state.global_step, total_optimizer_steps=4,
        collated_microbatches=collator.microbatches)
    assert audit['complete']
    assert audit['prefetched_microbatches'] == 0
    for step in (2, 4):
        validate_resume_checkpoint(tmp_path / 'checkpoints' / f'checkpoint-{step}',
                                   expected_step=step, expected_horizon=4)


def test_audit_rejects_missing_duplicate_and_wrong_second_order():
    ids = ['a', 'b', 'c']
    first = [{'epoch': 0, 'index': i, 'id': name, 'microbatch': i} for i, name in enumerate(ids)]
    kwargs = dict(seed=42, expected_epochs=2, global_step=4, total_optimizer_steps=4,
                  collated_microbatches=6)
    assert not sampling_audit(first, ids, **kwargs)['complete']
    second = [dict(r, epoch=1, microbatch=r['microbatch'] + 3) for r in first]
    assert not sampling_audit(first + second, ids, **kwargs)['complete']
    assert not sampling_audit(first + [first[0]], ids, **kwargs)['complete']


def test_smoke_longest_selection_keeps_frozen_order():
    from train_sorted_first import select_smoke_indices, validate_recipe
    rows = [{'length': n} for n in [3, 8, 2, 9, 7, 1]]
    assert select_smoke_indices(rows, lambda r: r['length'], count=3) == [1, 3, 4]
    with pytest.raises(ValueError, match='32'):
        select_smoke_indices(rows, lambda r: r['length'])
    cfg = {'model_path': '/opt/models/Qwen3-8B', 'init_adapter': None, 'seed': 42,
           'microbatch': 1, 'gradient_accumulation': 16, 'max_len': 4096,
           'r': 16, 'alpha': 32, 'dropout': .05, 'epochs': 5,
           'learning_rate': 1e-4, 'scheduler': 'cosine', 'warmup_ratio': .03,
           'targets': ['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj']}
    validate_recipe(cfg, rows=12828, total_steps=4010, save_steps=802, smoke=False)
    validate_recipe(cfg, rows=32, total_steps=4, save_steps=2, smoke=True)
    with pytest.raises(ValueError):
        validate_recipe(dict(cfg, microbatch=2), rows=12828, total_steps=4010,
                        save_steps=802, smoke=False)
    with pytest.raises(ValueError):
        validate_recipe(cfg, rows=12828, total_steps=4000, save_steps=802, smoke=False)
