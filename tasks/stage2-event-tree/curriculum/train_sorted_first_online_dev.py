"""Fresh Qwen3 LoRA, frozen Sorted1 course, resident full-dev feedback."""
import argparse
import json
import math
import os
from pathlib import Path
import shutil
import sys
import time

from online_dev_config import (HERE, jsonl, read, require, save, sha,
                               validate_manifest, verify_locks)


def smoke_dev_rows(rows, documents):
    """First two complete dev abstracts, never selected on scores."""
    document_ids = list(dict.fromkeys(r['document_id'] for r in rows))[:2]
    require(len(document_ids) == 2, 'Smoke requires two complete abstracts')
    selected = [r for r in rows if r['document_id'] in document_ids]
    gold = [d for d in documents if d['document_id'] in document_ids]
    require(len(gold) == 2 and {r['document_id'] for r in selected} ==
            {d['document_id'] for d in gold}, 'Smoke dev gold does not cover two abstracts')
    return selected, gold


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True, help='Exclusive training directory')
    p.add_argument('--manifest', type=Path, required=True)
    p.add_argument('--smoke', action='store_true')
    args = p.parse_args()
    require('TMUX' in os.environ, 'GPU training must run in tmux')
    os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, set_seed
    sys.path.insert(0, str(HERE.parent / 'scripts'))
    from model_utils import AssistantCollator, audit_rows, encode_training_row, load_model_config, provenance
    from online_dev_evaluation import evaluate_current_model
    from online_dev_trainer import OnlineDevTrainer, validate_online_checkpoint
    from sorted_first_sampler import IndexedDataset, ObservedCollator
    from train_sorted_first import select_smoke_indices
    from verify_sorted_first_online_dev import audit_sampling_prefix

    m = read(args.manifest)
    validate_manifest(m)
    require(torch.cuda.is_available() and torch.cuda.is_bf16_supported(), 'CUDA bf16 required')
    rows = jsonl(m['train_file'])
    dev = jsonl(Path(m['data_dir'])/'dev.jsonl')
    require(all(r['split'] == 'train' for r in rows), 'Non-train row in training data')
    require(all(r['split'] == 'dev' for r in dev), 'Non-dev row in evaluation data')
    set_seed(42)
    tokenizer = AutoTokenizer.from_pretrained(m['model_path'], local_files_only=True)
    tokenizer.padding_side = 'right'
    eval_tokenizer = AutoTokenizer.from_pretrained(m['model_path'], local_files_only=True)
    eval_tokenizer.padding_side = 'left'
    lengths = [len(encode_training_row(row, tokenizer)['input_ids']) for row in rows] if args.smoke else None
    indices = select_smoke_indices(list(enumerate(rows)), lambda item: lengths[item[0]]) if args.smoke else list(range(len(rows)))
    rows = [rows[i] for i in indices]
    token_audit = audit_rows(rows, tokenizer, m['max_len'])
    require(token_audit['overlength_count'] == 0, 'Truncation forbidden')
    horizon, cadence, epochs, warmup = (4, 2, 2, 1) if args.smoke else (3208, 401, 4, 97)
    args.output.mkdir(parents=True, exist_ok=False)
    save(args.output/'token_audit.json', token_audit)
    dev_file = Path(m['data_dir'])/'dev.jsonl'
    dev_document_file = Path(m['document_data_dir'])/'dev.jsonl'
    if args.smoke:
        dev, docs = smoke_dev_rows(dev, jsonl(dev_document_file))
        dev_file = args.output/'smoke-dev.jsonl'
        dev_document_file = args.output/'smoke-dev-documents.jsonl'
        for path, values in [(dev_file, dev), (dev_document_file, docs)]:
            path.write_text(''.join(json.dumps(v, ensure_ascii=False)+'\n' for v in values))
    config = {**m, **provenance(m['model_path'], Path(m['train_file'])),
              'max_steps': horizon, 'eval_steps': cadence, 'save_steps': cadence,
              'expected_epochs': epochs, 'warmup_steps': warmup, 'smoke': args.smoke,
              'epochs': epochs, 'steps_per_epoch': math.ceil(len(rows)/16),
              'ordered_ids': [r['id'] for r in rows], 'original_sorted_indices': indices,
              'source_token_lengths': lengths, 'dev_file': str(dev_file.resolve()),
              'dev_document_file': str(dev_document_file.resolve()), 'dev_count': len(dev),
              'dev_file_sha256': sha(dev_file), 'dev_document_sha256': sha(dev_document_file),
              'eval_strategy': 'steps', 'save_strategy': 'steps', 'resume_from_checkpoint': None,
              'load_best_model_at_end': False, 'optimizer': 'adamw_torch',
              'initialization': 'fresh base + new LoRA'}
    save(args.output/'config.json', config)
    started = time.monotonic()
    torch.cuda.reset_peak_memory_stats()
    model = AutoModelForCausalLM.from_pretrained(m['model_path'], local_files_only=True,
        config=load_model_config(m['model_path']), torch_dtype=torch.bfloat16, attn_implementation='sdpa')
    model.config.use_cache = False
    model = get_peft_model(model, LoraConfig(task_type='CAUSAL_LM', r=m['r'],
        lora_alpha=m['alpha'], lora_dropout=m['dropout'], target_modules=m['targets']))
    model.print_trainable_parameters()
    dataset = IndexedDataset([encode_training_row(r, tokenizer, m['max_len']) for r in rows],
                             [r['id'] for r in rows])
    training_args = TrainingArguments(output_dir=str(args.output/'checkpoints'),
        per_device_train_batch_size=1, gradient_accumulation_steps=16,
        learning_rate=.0001, lr_scheduler_type='constant', warmup_steps=warmup,
        adam_beta1=m['adam_beta1'], adam_beta2=m['adam_beta2'], adam_epsilon=m['adam_epsilon'],
        weight_decay=m['weight_decay'], num_train_epochs=epochs, max_steps=horizon,
        bf16=True, gradient_checkpointing=True, gradient_checkpointing_kwargs={'use_reentrant': False},
        seed=42, data_seed=42, logging_steps=1, logging_first_step=True,
        save_strategy='steps', save_steps=cadence, save_total_limit=None,
        eval_strategy='steps', eval_steps=cadence, eval_on_start=False,
        metric_for_best_model='AC_f1', greater_is_better=True, load_best_model_at_end=False,
        report_to='none', optim='adamw_torch', remove_unused_columns=False, label_names=['labels'],
        dataloader_num_workers=0, dataloader_drop_last=False)
    collator = ObservedCollator(AssistantCollator(tokenizer.pad_token_id))
    timings = []

    def evaluate_resident(current_model, step):
        # No test inputs or scores enter this closure.
        metrics = evaluate_current_model(current_model, eval_tokenizer, dev, dev_document_file,
            args.output/'online-dev'/f'dev-checkpoint-{step}', batch_size=4, max_new_tokens=319)
        timings.append(metrics['wall_seconds'])
        return metrics

    trainer = OnlineDevTrainer(model=model, args=training_args, train_dataset=dataset,
        eval_dataset=dev, data_collator=collator, processing_class=tokenizer,
        observation_path=args.output/'observed-order.jsonl', sampling_seed=42,
        dev_evaluator=evaluate_resident, feedback_path=args.output/'feedback.jsonl', warmup_steps=warmup)
    result = trainer.train()
    torch.cuda.synchronize()
    elapsed = time.monotonic()-started
    records = jsonl(args.output/'observed-order.jsonl')
    audit = audit_sampling_prefix(records, dataset.ids, global_step=trainer.state.global_step,
                                  max_epochs=epochs)
    save(args.output/'sampling-audit.json', audit)
    verify_locks(m['frozen_hashes'])
    feedback = trainer.feedback_controller.state_dict()
    step, best = trainer.state.global_step, feedback['best_step']
    require(best is not None, 'No complete online dev best')
    for checkpoint_step in range(cadence, step+1, cadence):
        validate_online_checkpoint(args.output/'checkpoints'/f'checkpoint-{checkpoint_step}', checkpoint_step, horizon)
    losses = [x['loss'] for x in trainer.state.log_history if 'loss' in x]
    finite = bool(losses) and all(math.isfinite(x) for x in losses) and math.isfinite(result.metrics['train_loss'])
    require(finite, 'Nonfinite/missing training loss')
    trainer.save_model(str(args.output/'last-adapter'))
    tokenizer.save_pretrained(args.output/'last-adapter')
    trainer.state.save_to_json(str(args.output/'trainer_state.json'))
    best_checkpoint = args.output/'checkpoints'/f'checkpoint-{best}'
    # Independent directory for deployable dev-best weights; complete checkpoint stays intact.
    (args.output/'best-adapter').mkdir(exist_ok=False)
    for filename in ('adapter_model.safetensors', 'adapter_config.json'):
        shutil.copy2(best_checkpoint/filename, args.output/'best-adapter'/filename)
    tokenizer.save_pretrained(args.output/'best-adapter')
    metrics = {**result.metrics, 'status': 'completed', 'smoke': args.smoke,
               'global_step': step, 'best_step': best, 'best_score': feedback['best_score'],
               'stop_reason': 'early_stop' if feedback['bad_count'] >= 6 else 'budget_exhausted',
               'finite_loss': finite, 'wall_seconds': elapsed, 'dev_wall_seconds': timings,
               'non_dev_seconds_per_update': (elapsed-sum(timings))/step,
               'peak_memory_bytes': torch.cuda.max_memory_allocated(), 'observed_examples': len(records),
               'observed_microbatches': trainer.observed_microbatches,
               'collated_microbatches': collator.microbatches, 'feedback': feedback}
    save(args.output/'train_metrics.json', metrics)
    print(json.dumps(metrics), flush=True)


if __name__ == '__main__':
    main()
