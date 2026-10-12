"""Train Action SFT from the completed CoT adapter with fresh Sorted1 state."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time

from action_common import MAX_NEW_TOKENS, DATA_DIR

ROOT = Path(__file__).resolve().parents[2]
CURRICULUM = ROOT / 'tasks/stage2-event-tree/curriculum'
DEFAULT_DATA = DATA_DIR
DEFAULT_COT = ROOT / 'tasks/type-cot-experiment/runs/C-CoT1-NoRE-GoldOrder-42-20261009/cot/final-adapter'


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def save(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def inspect_cot_source(adapter, model_path=None):
    adapter = Path(adapter).resolve()
    config = json.loads((adapter.parent / 'config.json').read_text())
    metrics = json.loads((adapter.parent / 'train_metrics.json').read_text())
    if metrics.get('status') != 'completed':
        raise ValueError('CoT initialization must be from completed training')
    if config.get('initialization') != 'fresh base + new LoRA':
        raise ValueError('Expected original fresh-base CoT initialization')
    if metrics.get('final_adapter_sha256') != sha(adapter / 'adapter_model.safetensors'):
        raise ValueError('CoT source adapter differs from completed training')
    if model_path is not None:
        adapter_config = json.loads((adapter / 'adapter_config.json').read_text())
        expected_base = Path(model_path).resolve()
        declared_base = adapter_config.get('base_model_name_or_path')
        if not declared_base or Path(declared_base).resolve() != expected_base:
            raise ValueError('CoT adapter base differs from requested SFT base model')
        if config.get('model_path') and Path(config['model_path']).resolve() != expected_base:
            raise ValueError('CoT training base differs from requested SFT base model')
    files = [adapter.parent / 'config.json', adapter.parent / 'train_metrics.json',
             adapter / 'adapter_config.json', adapter / 'adapter_model.safetensors']
    return {'status': 'completed', 'source_adapter': str(adapter),
            'source_sha256': {str(path): sha(path) for path in files}}


class TCFeedbackController:
    """Lexicographic TC/TI ranking; exact ties keep the earlier checkpoint."""
    def __init__(self, min_delta=.001):
        if not isinstance(min_delta, (int, float)) or isinstance(min_delta, bool) or not math.isfinite(min_delta) or min_delta <= 0:
            raise ValueError('min_delta must be a positive finite number')
        self.min_delta = min_delta
        self.anchor_tc = None
        self.best_score = None
        self.best_step = None
        self.bad_count = 0
        self.last_step = 0

    def update(self, step, metrics):
        if isinstance(step, bool) or not isinstance(step, int) or step <= self.last_step:
            raise ValueError('Evaluation steps must be increasing positive integers')
        scores = (metrics['TC_f1'], metrics['TI_f1'])
        if any(isinstance(x, bool) or not isinstance(x, (int, float)) or
               not math.isfinite(x) or not 0 <= x <= 1 for x in scores):
            raise ValueError('Presence TC/TI must be finite numbers in [0,1]')
        new_best = self.best_score is None or scores > self.best_score
        if new_best:
            self.best_score, self.best_step = scores, step
        if self.anchor_tc is None or scores[0] - self.anchor_tc >= self.min_delta - 1e-12:
            self.anchor_tc, self.bad_count = scores[0], 0
        else:
            self.bad_count += 1
        self.last_step = step
        return {**self.state_dict(), 'new_best': new_best,
                'reduce_lr': self.bad_count == 3, 'stop': self.bad_count >= 6}

    def state_dict(self):
        return {'best_score': self.best_score, 'best_step': self.best_step,
                'bad_count': self.bad_count, 'last_step': self.last_step,
                'anchor_tc': self.anchor_tc, 'min_delta': self.min_delta}


def assistant_token_loss(outputs, labels, num_items_in_batch=None):
    """Sum shifted assistant CE and normalize by this actual accumulation group.

    Trainer 4.52 supplies the group's unmasked-label count to custom loss
    functions and omits its fixed gradient_accumulation_steps division. User
    prefixes always contain masked tokens, so shifted and unshifted counts agree.
    This also normalizes the final 12-row accumulation group without /16 bias.
    """
    import torch.nn.functional as functional
    targets = labels[..., 1:]
    valid = targets != -100
    # Select supervised positions before FP32 conversion: prompt vocab logits
    # otherwise dominate the additional CE buffer with Qwen's large vocabulary.
    logits = outputs['logits'][..., :-1, :][valid].float()
    targets = targets[valid]
    denominator = valid.sum() if num_items_in_batch is None else num_items_in_batch
    if float(denominator) <= 0:
        raise ValueError('No assistant target tokens in accumulation group')
    return functional.cross_entropy(logits, targets, reduction='sum') / denominator


def make_trainer_class():
    """Lazy imports keep --help and validation available without GPU loading."""
    sys.path.insert(0, str(CURRICULUM))
    from sorted_first_sampler import SortedFirstTrainer
    from dev_feedback import WarmupPlateauScheduler

    class ActionSFTTrainer(SortedFirstTrainer):
        def __init__(self, *args, warmup_updates=97, **kwargs):
            self.warmup_updates = warmup_updates
            kwargs.setdefault('compute_loss_func', assistant_token_loss)
            super().__init__(*args, **kwargs)

        def create_scheduler(self, num_training_steps, optimizer=None):
            if self.lr_scheduler is None:
                self.lr_scheduler = WarmupPlateauScheduler(optimizer or self.optimizer,
                                                          self.warmup_updates)
                self._created_lr_scheduler = True
            return self.lr_scheduler

        def plateau_scheduler(self):
            scheduler = self.lr_scheduler
            while hasattr(scheduler, 'scheduler'):
                scheduler = scheduler.scheduler
            return scheduler

    return ActionSFTTrainer


def make_dev_callback(trainer, evaluator, output, tokenizer, checkpoint_steps,
                      controller=None, source=None):
    from transformers import TrainerCallback
    output = Path(output)
    controller = controller or TCFeedbackController()

    class ActionDevCallback(TrainerCallback):
        def on_train_begin(self, args, state, control, **kwargs):
            if state.global_step != 0 or trainer.optimizer.state_dict()['state']:
                raise ValueError('Action SFT requires a fresh optimizer at step zero')
            if trainer.plateau_scheduler().last_epoch != 0:
                raise ValueError('Action SFT requires a fresh LR scheduler')
            evidence = {'global_step': 0, 'optimizer_state_empty': True,
                        'scheduler_last_epoch': 0, 'resume_from_checkpoint': None}
            if source:
                from peft import get_peft_model_state_dict
                from safetensors.torch import load_file
                from curriculum_trainer import _equal
                expected = load_file(str(Path(source['source_adapter']) / 'adapter_model.safetensors'))
                if not _equal(expected, get_peft_model_state_dict(trainer.model)):
                    raise ValueError('Loaded adapter differs from CoT initialization')
                evidence['adapter_exact'] = True
            save(output / 'handoff_evidence.json', evidence)
            return control

        def on_step_end(self, args, state, control, **kwargs):
            if state.global_step not in checkpoint_steps:
                return control
            evaluation = evaluator(trainer.model, state.global_step)
            metrics = evaluation['metrics']
            decision = controller.update(state.global_step, metrics)
            with (output / 'feedback.jsonl').open('a') as stream:
                stream.write(json.dumps({'step': state.global_step, 'metrics': metrics,
                                         **decision}, allow_nan=False) + '\n')
            if decision['new_best']:
                trainer.save_model(str(output / 'best-adapter'))
                tokenizer.save_pretrained(output / 'best-adapter')
                save(output / 'best.json', {'step': state.global_step, 'metrics': metrics,
                                           'selection': 'presence TC, then TI, then earlier'})
            if decision['reduce_lr']:
                trainer.plateau_scheduler().reduce_lr()
            if decision['stop']:
                control.should_training_stop = True
            return control

    callback = ActionDevCallback()
    callback.feedback = controller
    return callback


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--data-dir', type=Path, default=DEFAULT_DATA)
    p.add_argument('--init-adapter', type=Path, default=DEFAULT_COT)
    p.add_argument('--model-path', default='/opt/models/Qwen3-8B')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--max-len', type=int, default=4096)
    p.add_argument('--max-new-tokens', type=int, default=MAX_NEW_TOKENS)
    p.add_argument('--dev-batch-size', type=int, default=4)
    p.add_argument('--smoke', action='store_true', help='Real Qwen training: 32 rows, 4 optimizer updates, 2 dev checks')
    return p


def main():
    args = parser().parse_args()
    if 'TMUX' not in os.environ:
        raise ValueError('GPU training must run inside tmux')
    if args.max_len < 1 or args.max_new_tokens < 1 or args.dev_batch_size < 1:
        raise ValueError('Token limits and batch size must be positive')
    if args.output.exists():
        raise FileExistsError(args.output)
    os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, set_seed
    sys.path.insert(0, str(CURRICULUM.parent / 'scripts'))
    sys.path.insert(0, str(CURRICULUM))
    from model_utils import AssistantCollator, audit_rows, encode_training_row, load_model_config
    from sorted_first_sampler import IndexedDataset, ObservedCollator
    from action_common import read_rows, verify_data, prepare_run, DEV_STEPS, verify_locks
    from action_evaluation import evaluate_model
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise ValueError('CUDA bf16 required')
    source = inspect_cot_source(args.init_adapter, args.model_path)
    verify_data(args.data_dir)
    rows = read_rows(args.data_dir / 'train-sorted.jsonl')
    dev = read_rows(args.data_dir / 'dev.jsonl')
    if any(row['split'] != 'train' for row in rows) or any(row['split'] != 'dev' for row in dev):
        raise ValueError('Training/dev split contamination')
    if args.smoke:
        rows = rows[:32]
        first_documents = list(dict.fromkeys(row['document_id'] for row in dev))[:2]
        dev = [row for row in dev if row['document_id'] in first_documents]
    steps_per_epoch = math.ceil(len(rows) / 16)
    epochs, horizon, warmup = (2, 4, 1) if args.smoke else (4, 4 * steps_per_epoch, 97)
    if not args.smoke and (len(rows) != 12828 or horizon != 3208):
        raise ValueError('Main Action SFT requires the complete 12,828-row Sorted1 split')
    checkpoint_steps = [2, 4] if args.smoke else list(DEV_STEPS)
    set_seed(args.seed)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, local_files_only=True)
    tokenizer.padding_side = 'right'
    eval_tokenizer = AutoTokenizer.from_pretrained(args.model_path, local_files_only=True)
    eval_tokenizer.padding_side = 'left'
    audit = audit_rows(rows, tokenizer, args.max_len)
    if audit['overlength_count']:
        raise ValueError('Action training exceeds max-len; truncation is forbidden')
    if not args.smoke and audit['max_assistant_tokens'] > args.max_new_tokens:
        raise ValueError('Generation limit cannot cover longest training target including EOS')
    preparation = prepare_run(args.output, 'action_sft', args.init_adapter,
        {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        model_path=args.model_path, allow_smoke=args.smoke)
    locks = preparation['input_sha256']
    save(args.output / 'token_audit.json', audit)
    config = {**vars(args), 'format': 'action-sft-v1', 'initialization': 'completed CoT adapter + fresh optimizer',
              'max_steps': horizon, 'epochs': epochs, 'warmup_steps': warmup,
              'microbatch': 1, 'gradient_accumulation': 16, 'learning_rate': 1e-4,
              'loss_normalization': 'all assistant tokens in actual accumulation group including tail',
              'checkpoint_steps': checkpoint_steps, 'selection_metric': 'presence_TC_f1',
              'selection_tiebreak': 'presence_TI_f1, then earlier', 'source_sha256': locks,
              'init_adapter': str(args.init_adapter.resolve()), 'source': source}
    save(args.output / 'config.json', {key: str(value) if isinstance(value, Path) else value
                                     for key, value in config.items()})
    started = time.monotonic()
    torch.cuda.reset_peak_memory_stats()
    model = AutoModelForCausalLM.from_pretrained(args.model_path, local_files_only=True,
        config=load_model_config(args.model_path), torch_dtype=torch.bfloat16, attn_implementation='sdpa')
    model.config.use_cache = False
    model = PeftModel.from_pretrained(model, str(args.init_adapter.resolve()),
                                      is_trainable=True, local_files_only=True)
    dataset = IndexedDataset([encode_training_row(row, tokenizer, args.max_len) for row in rows],
                             [row['id'] for row in rows])
    training_args = TrainingArguments(output_dir=str(args.output / 'checkpoints'),
        per_device_train_batch_size=1, gradient_accumulation_steps=16,
        learning_rate=1e-4, lr_scheduler_type='constant', warmup_steps=warmup,
        adam_beta1=.9, adam_beta2=.999, adam_epsilon=1e-8, weight_decay=0.,
        num_train_epochs=epochs, max_steps=horizon, bf16=True,
        gradient_checkpointing=True, gradient_checkpointing_kwargs={'use_reentrant': False},
        seed=args.seed, data_seed=args.seed, logging_steps=1, logging_first_step=True,
        save_strategy='no', eval_strategy='no', load_best_model_at_end=False,
        report_to='none', optim='adamw_torch', remove_unused_columns=False,
        label_names=['labels'], dataloader_num_workers=0, dataloader_drop_last=False)
    collator = ObservedCollator(AssistantCollator(tokenizer.pad_token_id))
    trainer = make_trainer_class()(model=model, args=training_args, train_dataset=dataset,
        data_collator=collator, processing_class=tokenizer,
        observation_path=args.output / 'observed-order.jsonl', sampling_seed=args.seed,
        warmup_updates=warmup)
    def evaluator(current_model, step):
        return evaluate_model(current_model, eval_tokenizer, dev,
            args.output / 'online-dev' / f'dev-checkpoint-{step}',
            batch_size=args.dev_batch_size, max_new_tokens=args.max_new_tokens)
    callback = make_dev_callback(trainer, evaluator, args.output, tokenizer,
                                 checkpoint_steps, source=source)
    trainer.add_callback(callback)
    result = trainer.train(resume_from_checkpoint=None)
    torch.cuda.synchronize()
    if callback.feedback.best_step is None:
        raise ValueError('No complete dev evaluation selected an adapter')
    trainer.save_model(str(args.output / 'last-adapter'))
    tokenizer.save_pretrained(args.output / 'last-adapter')
    trainer.state.save_to_json(str(args.output / 'trainer_state.json'))
    sys.path.insert(0, str(ROOT / 'tasks/type-cot-experiment/type-cot-goldorder-20261009'))
    from sampling import audit_sampling_prefix
    observed = [json.loads(line) for line in (args.output / 'observed-order.jsonl').read_text().splitlines()]
    sampling = audit_sampling_prefix(observed, dataset.ids,
        global_step=trainer.state.global_step, max_epochs=epochs, seed=args.seed)
    save(args.output / 'sampling-audit.json', sampling)
    if trainer.state.global_step not in checkpoint_steps:
        raise ValueError('Training stopped without a complete scheduled dev evaluation')
    if callback.feedback.bad_count < 6 and trainer.state.global_step != horizon:
        raise ValueError('Unexplained premature Action SFT stop')
    losses = [entry['loss'] for entry in trainer.state.log_history if 'loss' in entry]
    if (not losses or not all(math.isfinite(loss) for loss in losses)
            or not math.isfinite(result.metrics['train_loss'])):
        raise ValueError('Missing/nonfinite Action SFT training losses')
    verify_locks(locks)
    metrics = {**result.metrics, 'status': 'completed', 'stage': 'action_sft', 'smoke': args.smoke,
        'global_step': trainer.state.global_step, 'best_step': callback.feedback.best_step,
        'best_score': callback.feedback.best_score, 'feedback': callback.feedback.state_dict(),
        'finite_loss': True, 'wall_seconds': time.monotonic() - started,
        'peak_memory_bytes': torch.cuda.max_memory_allocated(),
        'best_adapter_sha256': sha(args.output / 'best-adapter/adapter_model.safetensors'),
        'best_adapter_config_sha256': sha(args.output / 'best-adapter/adapter_config.json'),
        'last_adapter_sha256': sha(args.output / 'last-adapter/adapter_model.safetensors'),
        'stop_reason': 'early_stop' if callback.feedback.bad_count >= 6 else 'budget_exhausted'}
    save(args.output / 'train_metrics.json', metrics)
    print(json.dumps(metrics, allow_nan=False), flush=True)


if __name__ == '__main__':
    main()
