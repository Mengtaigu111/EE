"""One raw-Qwen3 AppendixNoCopy run: sorted first epoch, shuffled later epochs."""
import argparse
import json
import math
import os
from pathlib import Path
import sys
import time

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'scripts'))
from model_utils import (AssistantCollator, audit_rows, encode_training_row, file_hash,
                         load_model_config, provenance, read_rows, write_json)


def select_smoke_indices(rows, length, count=32):
    """Longest real examples, stable tie break, then restore frozen difficulty order."""
    if len(rows) < count:
        raise ValueError(f'Smoke requires at least {count} rows')
    return sorted(sorted(range(len(rows)), key=lambda i: (-length(rows[i]), i))[:count])


def validate_recipe(cfg, *, rows, total_steps, save_steps, smoke):
    required = {'model_path': '/opt/models/Qwen3-8B', 'init_adapter': None, 'seed': 42,
        'microbatch': 1, 'gradient_accumulation': 16, 'max_len': 4096,
        'r': 16, 'alpha': 32, 'dropout': .05, 'epochs': 5,
        'learning_rate': .0001, 'scheduler': 'cosine', 'warmup_ratio': .03,
        'targets': ['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj']}
    for key, expected in required.items():
        if cfg.get(key) != expected:
            raise ValueError(f'Raw baseline recipe mismatch: {key}')
    expected = (32, 4, 2) if smoke else (12828, 4010, 802)
    if (rows, total_steps, save_steps) != expected:
        raise ValueError(f'Wrong rows/update/checkpoint budget: expected {expected}')
    epochs = 2 if smoke else 5
    if math.ceil(rows / cfg['gradient_accumulation']) * epochs != total_steps:
        raise ValueError('Budget does not span exactly the required complete epochs')
    return epochs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--baseline', type=Path, required=True)
    parser.add_argument('--total-steps', type=int, required=True)
    parser.add_argument('--save-steps', type=int, required=True)
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    if 'TMUX' not in os.environ:
        raise RuntimeError('Run GPU training in tmux')
    os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, set_seed
    from curriculum_trainer import validate_resume_checkpoint
    from sorted_first_sampler import IndexedDataset, ObservedCollator, SortedFirstTrainer, sampling_audit
    cfg_path = args.baseline / 'training/config.json'
    cfg = json.loads(cfg_path.read_text())
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError('CUDA bf16 required')
    rows = read_rows(args.data)
    if any(row.get('mode') != 'sentence-appendix-schema' for row in rows):
        raise ValueError('Frozen original AppendixNoCopy rows required')
    if len({row['id'] for row in rows}) != len(rows):
        raise ValueError('Duplicate train row IDs')
    # Validate baseline/budget before allocating the base model or writing run files.
    expected_epochs = validate_recipe(cfg, rows=32 if args.smoke else len(rows),
        total_steps=args.total_steps, save_steps=args.save_steps, smoke=args.smoke)
    set_seed(cfg['seed'])
    tokenizer = AutoTokenizer.from_pretrained(cfg['model_path'], local_files_only=True)
    tokenizer.padding_side = 'right'
    source_token_lengths = [len(encode_training_row(row, tokenizer)['input_ids'])
                            for row in rows] if args.smoke else None
    indices = select_smoke_indices(list(enumerate(rows)),
        lambda item: source_token_lengths[item[0]]) if args.smoke else list(range(len(rows)))
    rows = [rows[i] for i in indices]
    token_audit = audit_rows(rows, tokenizer, cfg['max_len'])
    if token_audit['overlength_count']:
        raise ValueError('Truncation forbidden')
    args.output.mkdir(parents=True, exist_ok=False)
    write_json(args.output / 'token_audit.json', token_audit)
    source_paths = [Path(__file__).resolve(), HERE / 'sorted_first_sampler.py',
                    HERE / 'curriculum_trainer.py', HERE.parent / 'scripts/model_utils.py',
                    cfg_path.resolve(), args.data.resolve()]
    source_hashes = {str(path): file_hash(path) for path in source_paths}
    config = {**cfg, **provenance(cfg['model_path'], args.data), 'baseline': str(args.baseline.resolve()),
        'max_steps': args.total_steps, 'scheduler_horizon': args.total_steps,
        'save_strategy': 'steps', 'save_steps': args.save_steps, 'smoke': args.smoke,
        'expected_epochs': expected_epochs, 'steps_per_epoch': math.ceil(len(rows) / 16),
        'ordered_ids': [row['id'] for row in rows], 'original_sorted_indices': indices,
        'source_token_lengths': source_token_lengths,
        'source_hashes': source_hashes, 'curriculum_script_hashes':
            {p.name: file_hash(p) for p in HERE.glob('*.py')},
        'sampling': 'epoch0 sequential; epoch>=1 torch randperm(seed+epoch)',
        'observation': 'actual training_step, epoch carried through sampler index',
        'state_format': 'safetensors + JSON, no pickle',
        'checkpoint_selection': 'full dev official AC_f1, later-step tie break',
        'resume_from_checkpoint': None, 'continuous_trainer': True}
    write_json(args.output / 'config.json', config)
    started = time.time()
    torch.cuda.reset_peak_memory_stats()
    model = AutoModelForCausalLM.from_pretrained(cfg['model_path'],
        config=load_model_config(cfg['model_path']), local_files_only=True,
        torch_dtype=torch.bfloat16, attn_implementation='sdpa')
    model.config.use_cache = False
    model = get_peft_model(model, LoraConfig(task_type='CAUSAL_LM', r=cfg['r'],
        lora_alpha=cfg['alpha'], lora_dropout=cfg['dropout'], target_modules=cfg['targets']))
    model.print_trainable_parameters()
    dataset = IndexedDataset([encode_training_row(row, tokenizer, cfg['max_len']) for row in rows],
                             [row['id'] for row in rows])
    training_args = TrainingArguments(output_dir=str(args.output / 'checkpoints'),
        per_device_train_batch_size=cfg['microbatch'], gradient_accumulation_steps=cfg['gradient_accumulation'],
        learning_rate=cfg['learning_rate'], lr_scheduler_type=cfg['scheduler'], warmup_ratio=cfg['warmup_ratio'],
        num_train_epochs=cfg['epochs'], max_steps=args.total_steps, bf16=True,
        gradient_checkpointing=True, gradient_checkpointing_kwargs={'use_reentrant': False},
        seed=cfg['seed'], data_seed=cfg['seed'], logging_steps=1, logging_first_step=True,
        save_strategy='steps', save_steps=args.save_steps, save_total_limit=None, eval_strategy='no',
        report_to='none', optim='adamw_torch', remove_unused_columns=False, label_names=['labels'],
        dataloader_num_workers=0, dataloader_drop_last=False)
    collator = ObservedCollator(AssistantCollator(tokenizer.pad_token_id))
    observed_path = args.output / 'observed-order.jsonl'
    trainer = SortedFirstTrainer(model=model, args=training_args, train_dataset=dataset,
        data_collator=collator, processing_class=tokenizer,
        observation_path=observed_path, sampling_seed=cfg['seed'])
    result = trainer.train()
    torch.cuda.synchronize()
    wall_seconds = time.time() - started
    records = [json.loads(line) for line in observed_path.read_text().splitlines()]
    audit = sampling_audit(records, dataset.ids, seed=cfg['seed'], expected_epochs=expected_epochs,
        global_step=trainer.state.global_step, total_optimizer_steps=args.total_steps,
        collated_microbatches=collator.microbatches)
    write_json(args.output / 'sampling-audit.json', audit)
    if not audit['complete']:
        raise RuntimeError('Observed sampling failed full sorted-first epoch coverage audit')
    if any(file_hash(path) != expected for path, expected in source_hashes.items()):
        raise RuntimeError('Training source changed during this run')
    for step in range(args.save_steps, args.total_steps + 1, args.save_steps):
        validate_resume_checkpoint(args.output / 'checkpoints' / f'checkpoint-{step}',
            expected_step=step, expected_horizon=args.total_steps)
    losses = [x['loss'] for x in trainer.state.log_history if 'loss' in x]
    finite_loss = bool(losses) and all(math.isfinite(loss) for loss in losses) and math.isfinite(result.metrics['train_loss'])
    if not finite_loss:
        raise FloatingPointError('Nonfinite or missing training losses')
    trainer.save_model(str(args.output / 'adapter'))
    tokenizer.save_pretrained(args.output / 'adapter')
    trainer.state.save_to_json(str(args.output / 'trainer_state.json'))
    metrics = {**result.metrics, 'global_step': trainer.state.global_step,
        'phase_optimizer_steps': trainer.state.global_step, 'total_optimizer_steps': args.total_steps,
        'scheduler_horizon': args.total_steps, 'finite_loss': finite_loss,
        'first_logged_loss': losses[0], 'last_logged_loss': losses[-1],
        'phase_mean_logged_loss': sum(losses) / len(losses),
        'wall_seconds': wall_seconds, 'seconds_per_update': wall_seconds / trainer.state.global_step,
        'peak_memory_bytes': torch.cuda.max_memory_allocated(), 'phase_rows': len(rows),
        'collated_examples': collator.examples, 'collated_batches': collator.microbatches,
        'collated_assistant_tokens': collator.assistant_tokens,
        'observed_examples': len(records), 'observed_microbatches': trainer.observed_microbatches,
        'sampling_audit_complete': audit['complete'], 'source_hashes': source_hashes,
        'smoke': args.smoke, 'status': 'completed'}
    write_json(args.output / 'train_metrics.json', metrics)
    print(json.dumps(metrics), flush=True)


if __name__ == '__main__':
    main()
