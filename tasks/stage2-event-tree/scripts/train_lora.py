"""Train the Stage-2 bf16 LoRA adapter (fixed final epoch); --audit-only never loads weights."""
import argparse
import json
import os
import time
from pathlib import Path

from model_utils import (DEFAULT_MODEL, AssistantCollator, audit_rows,
                         encode_training_row, file_hash, load_model_config, provenance,
                         read_rows, write_json)

TARGETS = ['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj']


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--model', default=DEFAULT_MODEL)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--max-len', type=int, default=2048)
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--learning-rate', type=float, default=1e-4)
    parser.add_argument('--max-steps', type=int, default=-1)
    parser.add_argument('--limit', type=int)
    parser.add_argument('--init-adapter', type=Path, help='continue training from an existing LoRA adapter (curriculum / warm start)')
    parser.add_argument('--audit-only', action='store_true')
    args = parser.parse_args()
    if args.max_len < 1 or args.max_steps == 0 or args.max_steps < -1 or args.epochs < 1:
        parser.error('max-len/epochs must be positive and max-steps positive or -1')
    os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    tokenizer.padding_side = 'right'
    rows = read_rows(args.data, args.limit)
    audit = audit_rows(rows, tokenizer, args.max_len)
    if args.audit_only:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        write_json(args.output, {**audit, **provenance(args.model, args.data)})
        print(json.dumps(audit), flush=True)
        return
    args.output.mkdir(parents=True, exist_ok=False)
    write_json(args.output / 'token_audit.json', audit)
    if audit['overlength_count']:
        raise ValueError(f"{audit['overlength_count']} overlength examples: see token_audit.json; no data truncated")
    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, Trainer, TrainingArguments, set_seed
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError('A CUDA GPU supporting bf16 is required')
    set_seed(args.seed)
    torch.cuda.reset_peak_memory_stats()
    config = {**provenance(args.model, args.data), 'seed': args.seed, 'mode': rows[0].get('mode'),
              'max_len': args.max_len, 'max_steps': args.max_steps, 'limit': args.limit,
              'precision': 'bf16', 'quantization': None, 'r': 16, 'alpha': 32, 'dropout': .05, 'targets': TARGETS,
              'learning_rate': args.learning_rate, 'scheduler': 'cosine', 'warmup_ratio': .03,
              'epochs': args.epochs, 'microbatch': 1, 'gradient_accumulation': 16,
              'gradient_checkpointing': True, 'enable_thinking': False,
              'loss': 'assistant tokens only', 'checkpoint_selection': 'fixed final epoch; all epoch checkpoints kept',
              'init_adapter': str(args.init_adapter.resolve()) if args.init_adapter else None,
              'init_adapter_sha256': file_hash(args.init_adapter / 'adapter_model.safetensors') if args.init_adapter else None,
              'gpu': torch.cuda.get_device_name(), 'cuda': torch.version.cuda}
    write_json(args.output / 'config.json', config)
    started = time.time()
    model = AutoModelForCausalLM.from_pretrained(args.model, config=load_model_config(args.model),
        local_files_only=True, torch_dtype=torch.bfloat16, attn_implementation='sdpa')
    model.config.use_cache = False
    if args.init_adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, args.init_adapter, is_trainable=True)
    else:
        model = get_peft_model(model, LoraConfig(task_type='CAUSAL_LM', r=16, lora_alpha=32,
                                                 lora_dropout=.05, target_modules=TARGETS))
    model.print_trainable_parameters()
    dataset = [encode_training_row(row, tokenizer, args.max_len) for row in rows]
    training_args = TrainingArguments(output_dir=str(args.output / 'checkpoints'),
        per_device_train_batch_size=1, gradient_accumulation_steps=16,
        learning_rate=args.learning_rate, lr_scheduler_type='cosine', warmup_ratio=.03,
        num_train_epochs=args.epochs, max_steps=args.max_steps, bf16=True,
        gradient_checkpointing=True, gradient_checkpointing_kwargs={'use_reentrant': False},
        seed=args.seed, data_seed=args.seed, logging_steps=1, logging_first_step=True,
        save_strategy='epoch', save_total_limit=None, eval_strategy='no',
        report_to='none', optim='adamw_torch', remove_unused_columns=False, label_names=['labels'])
    trainer = Trainer(model=model, args=training_args, train_dataset=dataset,
                      data_collator=AssistantCollator(tokenizer.pad_token_id), processing_class=tokenizer)
    result = trainer.train()
    trainer.save_model(str(args.output / 'adapter'))
    tokenizer.save_pretrained(args.output / 'adapter')
    trainer.state.save_to_json(str(args.output / 'trainer_state.json'))
    metrics = {**result.metrics, 'wall_seconds': time.time() - started,
               'peak_memory_bytes': torch.cuda.max_memory_allocated(),
               'global_step': trainer.state.global_step, 'epoch': trainer.state.epoch,
               'smoke': args.max_steps > 0 or args.limit is not None, 'status': 'completed'}
    write_json(args.output / 'train_metrics.json', metrics)
    print(json.dumps(metrics), flush=True)


if __name__ == '__main__':
    main()
