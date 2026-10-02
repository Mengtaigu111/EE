"""One phase of the baseline C recipe, with a shared global training horizon."""
import argparse
import json
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'scripts'))
from model_utils import (AssistantCollator, audit_rows, encode_training_row,
                         load_model_config, provenance, read_rows, write_json)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--baseline', type=Path, required=True)
    p.add_argument('--total-steps', type=int, required=True)
    p.add_argument('--stop-step', type=int, required=True)
    p.add_argument('--save-steps', type=int, required=True)
    p.add_argument('--resume', type=Path)
    p.add_argument('--limit', type=int)
    p.add_argument('--smoke', action='store_true')
    a = p.parse_args()
    if 'TMUX' not in os.environ:
        raise RuntimeError('Run GPU training in tmux')
    if not 0 < a.stop_step <= a.total_steps or a.save_steps < 1:
        p.error('invalid global step budget')
    os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, set_seed
    from curriculum_trainer import CurriculumTrainer, PhaseStopCallback, validate_resume_checkpoint
    cfg = json.loads((a.baseline / 'training/config.json').read_text())
    model_path, seed = cfg['model_path'], cfg['seed']
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError('CUDA bf16 required')
    if cfg['init_adapter'] is not None:
        raise ValueError('Baseline must initialize from raw base model')
    start_step = 0
    if a.resume:
        prior = json.loads((a.resume / 'trainer_state.json').read_text())
        start_step = prior['global_step']
        validate_resume_checkpoint(a.resume, expected_step=start_step, expected_horizon=a.total_steps)
    if start_step >= a.stop_step:
        raise ValueError('Phase has no remaining updates')
    a.output.mkdir(parents=True, exist_ok=False)
    set_seed(seed)
    torch.cuda.reset_peak_memory_stats()
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    tokenizer.padding_side = 'right'
    rows = read_rows(a.data)
    if a.limit:
        # Exercise the longest real samples for smoke; full runs preserve all rows.
        rows = sorted(rows, key=lambda r: len(encode_training_row(r, tokenizer)['input_ids']), reverse=True)[:a.limit]
    audit = audit_rows(rows, tokenizer, cfg['max_len'])
    write_json(a.output / 'token_audit.json', audit)
    if audit['overlength_count']:
        raise ValueError('Truncation forbidden')
    write_json(a.output / 'config.json', {
        **cfg, **provenance(model_path, a.data), 'baseline': str(a.baseline.resolve()),
        'max_steps': a.total_steps, 'phase_start_step': start_step, 'phase_stop_step': a.stop_step,
        'resume_from_checkpoint': str(a.resume.resolve()) if a.resume else None,
        'ignore_data_skip': bool(a.resume), 'save_strategy': 'steps', 'save_steps': a.save_steps,
        'scheduler_horizon': a.total_steps, 'smoke': a.smoke,
        'checkpoint_selection': 'full dev official AC_f1, later-step tie break',
        'state_format': 'safetensors + JSON, no pickle',
        'curriculum_script_hashes': {x.name: __import__('hashlib').sha256(x.read_bytes()).hexdigest()
                                     for x in HERE.glob('*.py')}})
    started = time.time()
    model = AutoModelForCausalLM.from_pretrained(model_path, config=load_model_config(model_path),
        local_files_only=True, torch_dtype=torch.bfloat16, attn_implementation='sdpa')
    model.config.use_cache = False
    model = get_peft_model(model, LoraConfig(task_type='CAUSAL_LM', r=cfg['r'], lora_alpha=cfg['alpha'],
        lora_dropout=cfg['dropout'], target_modules=cfg['targets']))
    model.print_trainable_parameters()
    dataset = [encode_training_row(r, tokenizer, cfg['max_len']) for r in rows]
    training_args = TrainingArguments(output_dir=str(a.output / 'checkpoints'),
        per_device_train_batch_size=cfg['microbatch'], gradient_accumulation_steps=cfg['gradient_accumulation'],
        learning_rate=cfg['learning_rate'], lr_scheduler_type=cfg['scheduler'], warmup_ratio=cfg['warmup_ratio'],
        num_train_epochs=cfg['epochs'], max_steps=a.total_steps, bf16=True,
        gradient_checkpointing=True, gradient_checkpointing_kwargs={'use_reentrant': False},
        seed=seed, data_seed=seed, logging_steps=1, logging_first_step=True,
        save_strategy='steps', save_steps=a.save_steps, save_total_limit=None, eval_strategy='no',
        ignore_data_skip=bool(a.resume), report_to='none', optim='adamw_torch',
        remove_unused_columns=False, label_names=['labels'])
    consumed = {'examples': 0, 'batches': 0, 'assistant_tokens': 0}
    base_collator = AssistantCollator(tokenizer.pad_token_id)
    def counting_collator(batch):
        consumed['examples'] += len(batch)
        consumed['batches'] += 1
        consumed['assistant_tokens'] += sum(sum(x != -100 for x in row['labels']) for row in batch)
        return base_collator(batch)
    trainer = CurriculumTrainer(model=model, args=training_args, train_dataset=dataset,
        data_collator=counting_collator, processing_class=tokenizer,
        callbacks=[PhaseStopCallback(a.stop_step)])
    result = trainer.train(resume_from_checkpoint=str(a.resume) if a.resume else None)
    if trainer.state.global_step != a.stop_step:
        raise RuntimeError(f'Wrong end step: {trainer.state.global_step}')
    checkpoint = a.output / 'checkpoints' / f'checkpoint-{a.stop_step}'
    validate_resume_checkpoint(checkpoint, expected_step=a.stop_step, expected_horizon=a.total_steps)
    trainer.save_model(str(a.output / 'adapter'))
    tokenizer.save_pretrained(a.output / 'adapter')
    trainer.state.save_to_json(str(a.output / 'trainer_state.json'))
    if a.resume:
        write_json(a.output / 'resume_evidence.json', trainer.resume_evidence)
    updates = a.stop_step - start_step
    losses = [x for x in trainer.state.log_history if start_step < x.get('step', 0) <= a.stop_step and 'loss' in x]
    metrics = {**result.metrics, 'phase_train_loss': result.metrics['train_loss'] * a.stop_step / updates,
        'phase_mean_logged_loss': sum(x['loss'] for x in losses) / len(losses) if losses else None,
        'first_logged_loss': losses[0]['loss'] if losses else None,
        'last_logged_loss': losses[-1]['loss'] if losses else None,
        'phase_optimizer_steps': updates, 'phase_start_step': start_step, 'global_step': trainer.state.global_step,
        'scheduler_horizon': a.total_steps, 'checkpoint': str(checkpoint),
        'wall_seconds': time.time() - started, 'peak_memory_bytes': torch.cuda.max_memory_allocated(),
        'phase_rows': len(rows), 'collated_examples': consumed['examples'],
        'collated_batches': consumed['batches'], 'collated_assistant_tokens': consumed['assistant_tokens'],
        'effective_collated_passes': consumed['examples'] / len(rows),
        'sampling_note': 'Fresh shuffled iterator; collator count may include one prefetched batch; Trainer epoch is global bookkeeping after dataset switch',
        'smoke': a.smoke, 'status': 'completed'}
    write_json(a.output / 'train_metrics.json', metrics)
    print(json.dumps(metrics), flush=True)


if __name__ == '__main__':
    main()
