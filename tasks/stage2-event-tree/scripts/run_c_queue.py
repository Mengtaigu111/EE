#!/usr/bin/env python3
"""Run Experiment C with sentence-level inference and dev-best checkpoint selection."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / 'scripts'
DEFAULT_SENTENCE_DATA = ROOT / 'data/sentence-c'
DEFAULT_DOCUMENT_DATA = ROOT / 'data/none'


def now():
    return datetime.now(timezone.utc).isoformat()


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    temporary.replace(path)


def run_stage(command, log_path):
    if log_path.exists():
        raise FileExistsError(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open('x', encoding='utf-8') as stream:
        stream.write(shlex.join(command) + '\n')
        stream.flush()
        started = time.time()
        result = subprocess.run(command, cwd=ROOT.parents[1], stdout=stream, stderr=subprocess.STDOUT)
    if result.returncode:
        raise subprocess.CalledProcessError(result.returncode, command)
    return time.time() - started


def generation_budget(audit, explicit=None):
    if explicit is not None:
        if explicit < 1:
            raise ValueError('max_new_tokens must be positive')
        return explicit
    # The target length is re-audited for C; this is deliberately independent of T0's 768.
    return max(64, int(audit['max_assistant_tokens'] * 1.25) + 16)


def checkpoint_key(path):
    match = re.fullmatch(r'checkpoint-(\d+)', path.name)
    return int(match.group(1)) if match else -1


def read_metrics(path):
    return json.loads(path.read_text(encoding='utf-8'))


def evaluate_checkpoint(adapter, split, args, run_dir, budget):
    prefix = run_dir / 'checkpoint-eval' / f'{split}-{adapter.name}'
    prediction = prefix.with_suffix('.predictions.jsonl')
    merged = prefix.with_suffix('.merged.jsonl')
    metrics = prefix.with_suffix('.metrics.json')
    run_stage([sys.executable, str(SCRIPTS / 'predict.py'), '--data', str(args.data_dir / f'{split}.jsonl'),
               '--output', str(prediction), '--model', args.model, '--adapter', str(adapter),
               '--batch-size', str(args.batch_size), '--seed', str(args.seed),
               '--max-new-tokens', str(budget)], prefix.with_suffix('.predict.log'))
    run_stage([sys.executable, str(SCRIPTS / 'merge_sentence_predictions.py'), '--data',
               str(args.data_dir / f'{split}.jsonl'), '--predictions', str(prediction),
               '--output', str(merged)], prefix.with_suffix('.merge.log'))
    run_stage([sys.executable, str(SCRIPTS / 'evaluate.py'), '--data',
               str(args.document_data_dir / f'{split}.jsonl'), '--predictions', str(merged),
               '--output', str(metrics)] + (['--complex'] if split == 'test' else []),
              prefix.with_suffix('.evaluate.log'))
    return read_metrics(metrics), prediction, merged


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, default=DEFAULT_SENTENCE_DATA)
    parser.add_argument('--document-data-dir', type=Path, default=DEFAULT_DOCUMENT_DATA)
    parser.add_argument('--run-dir', type=Path)
    parser.add_argument('--model', default='/opt/models/Qwen3-8B')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--max-len', type=int, default=4096)
    parser.add_argument('--max-steps', type=int, default=-1,
                        help='explicit optimizer-step budget; -1 uses all epochs')
    parser.add_argument('--max-new-tokens', type=int,
                        help='override the C sentence-level audit-derived generation budget')
    parser.add_argument('--batch-size', type=int, default=4)
    args = parser.parse_args()
    if 'TMUX' not in os.environ:
        raise RuntimeError('GPU execution must run in tmux')
    if args.epochs < 1 or args.max_len < 1 or args.max_steps == 0 or args.max_steps < -1:
        parser.error('epochs/max-len must be positive and max-steps must be -1 or positive')
    if args.run_dir is None:
        stamp = datetime.now().strftime('%Y%m%d-%H%M%S')
        args.run_dir = ROOT / 'runs' / f'C-{args.seed}-{stamp}'
    args.run_dir = args.run_dir.resolve()
    args.run_dir.mkdir(parents=True, exist_ok=False)
    state = {'experiment': 'C', 'seed': args.seed, 'status': 'running', 'started_at': now(),
             'data_dir': str(args.data_dir.resolve()), 'document_data_dir': str(args.document_data_dir.resolve()),
             'epochs': args.epochs, 'max_steps': args.max_steps, 'max_len': args.max_len,
             'model': args.model}
    save(args.run_dir / 'run.json', state)
    audit_path = args.run_dir / 'sentence-train.audit.json'
    run_stage([sys.executable, str(SCRIPTS / 'train_lora.py'), '--data',
               str(args.data_dir / 'train.jsonl'), '--output', str(audit_path), '--model', args.model,
               '--max-len', str(args.max_len), '--audit-only'], args.run_dir / 'audit.log')
    audit = read_metrics(audit_path)
    budget = generation_budget(audit, args.max_new_tokens)
    state.update({'audit': audit, 'max_new_tokens': budget})
    save(args.run_dir / 'run.json', state)
    training_dir = args.run_dir / 'training'
    train_command = [sys.executable, str(SCRIPTS / 'train_lora.py'), '--data',
                     str(args.data_dir / 'train.jsonl'), '--output', str(training_dir), '--model', args.model,
                     '--seed', str(args.seed), '--epochs', str(args.epochs), '--max-len', str(args.max_len),
                     '--max-steps', str(args.max_steps)]
    run_stage(train_command, args.run_dir / 'train.log')
    checkpoints = sorted((training_dir / 'checkpoints').glob('checkpoint-*'), key=checkpoint_key)
    if not checkpoints:
        checkpoints = [training_dir / 'adapter']
    dev_results = []
    for checkpoint in checkpoints:
        metrics, _, _ = evaluate_checkpoint(checkpoint, 'dev', args, args.run_dir, budget)
        dev_results.append({'checkpoint': str(checkpoint), 'step': checkpoint_key(checkpoint),
                            'AC_f1': metrics['official']['AC_f1'], 'metrics': str(
                                (args.run_dir / 'checkpoint-eval' / f'dev-{checkpoint.name}.metrics.json'))})
    best = max(dev_results, key=lambda result: (result['AC_f1'], result['step']))
    best_adapter = Path(best['checkpoint'])
    test_metrics, test_prediction, test_merged = evaluate_checkpoint(best_adapter, 'test', args, args.run_dir, budget)
    state.update({'status': 'completed', 'finished_at': now(), 'best_checkpoint': str(best_adapter),
                  'dev_checkpoints': dev_results, 'test_metrics': str(
                      args.run_dir / 'checkpoint-eval' / f'test-{best_adapter.name}.metrics.json'),
                  'test_prediction': str(test_prediction), 'test_merged': str(test_merged),
                  'test_official': test_metrics['official']})
    save(args.run_dir / 'run.json', state)
    print(json.dumps({'run_dir': str(args.run_dir), 'audit': audit, 'max_new_tokens': budget,
                      'best_checkpoint': str(best_adapter), 'test_official': test_metrics['official']},
                     ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
