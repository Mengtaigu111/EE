"""Predict with repaired parser, merge with sentence scope, use official scores."""
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
FIXED = HERE.parent / 'evaluation-v2-20261003/scripts'
sys.path.insert(0, str(HERE.parent / 'scripts'))
from run_c_queue import run_stage

COMPLEX_KEYS = {'Consecutive_AC_f1', 'Overlap_TC_f1', 'Overlap_AC_f1',
                'Reverse_AC_f1', 'Subevent_TC_f1', 'Subevent_AC_f1', 'Subevent_EC_f1'}

def evaluate_checkpoint(adapter, split, args, run_dir, budget):
    prefix = run_dir / 'checkpoint-eval' / f'{split}-{adapter.name}'
    prediction = prefix.with_suffix('.predictions.jsonl')
    merged = prefix.with_suffix('.merged.jsonl')
    metrics = prefix.with_suffix('.metrics.json')
    run_stage([sys.executable, str(HERE / 'predict_appendix_fixed.py'), '--data',
               str(args.data_dir / f'{split}.jsonl'), '--output', str(prediction),
               '--model', args.model, '--adapter', str(adapter), '--batch-size', str(args.batch_size),
               '--seed', str(args.seed), '--max-new-tokens', str(budget)], prefix.with_suffix('.predict.log'))
    run_stage([sys.executable, str(FIXED / 'merge_sentence_predictions.py'), '--data',
               str(args.data_dir / f'{split}.jsonl'), '--predictions', str(prediction),
               '--output', str(merged)], prefix.with_suffix('.merge.log'))
    run_stage([sys.executable, str(FIXED / 'evaluate.py'), '--data',
               str(args.document_data_dir / f'{split}.jsonl'), '--predictions', str(merged),
               '--output', str(metrics)] + (['--complex'] if split == 'test' else []),
              prefix.with_suffix('.evaluate.log'))
    values = json.loads(metrics.read_text())
    if split == 'test':
        complex_scores = values.get('complex', {})
        if complex_scores.get('exit_code') != 0 or not COMPLEX_KEYS.issubset(complex_scores.get('scores', {})):
            raise ValueError('Test complex evaluation must succeed with all seven scores')
    return values, prediction, merged
