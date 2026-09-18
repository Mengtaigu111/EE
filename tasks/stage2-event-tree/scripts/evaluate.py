"""Align generated forests to offsets, write SciEvents-format predictions, score with the official metric.

Main numbers come from utils.calculate_F1 (EXCEEDS repo, offset expression). For the test split
the official complex-scenario script (eval_complex_quick_f1.py) is also run as a subprocess,
because it hard-codes data/SciEvents/test.json as gold.
"""
import argparse
from collections import Counter
import json
from pathlib import Path
import re
import subprocess
import sys

from align_events import align_forest
from ontology import EVENT_TYPES

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))
import utils  # noqa: E402  (EXCEEDS official scorer)

METRIC_NAMES = ['TI_f1', 'TC_f1', 'AI_f1', 'AC_f1', 'EC_I_f1', 'EC_C_f1',
                'TI_p', 'TC_p', 'AI_p', 'AC_p', 'EC_I_p', 'EC_C_p',
                'TI_r', 'TC_r', 'AI_r', 'AC_r', 'EC_I_r', 'EC_C_r']


def read_jsonl(path):
    with open(path, encoding='utf-8') as stream:
        return [json.loads(line) for line in stream if line.strip()]


def official_scores(golds, preds):
    values = utils.calculate_F1(golds, preds, 'offset')
    return {name: float(value) for name, value in zip(METRIC_NAMES, values)}


def strip_private(events):
    return [{'event_type': e['event_type'],
             'trigger': {k: e['trigger'][k] for k in ('text', 'tokens', 'offsets')},
             'arguments': [{k: a[k] for k in ('text', 'tokens', 'offsets', 'argument_type')} for a in e['arguments']]}
            for e in events]


def evaluate(data_rows, predictions, oracle=False):
    by_id = {row['id']: row for row in data_rows}
    if len(by_id) != len(data_rows):
        raise ValueError('Duplicate data IDs')
    if oracle:
        from linearize import canonical_events
        predictions = [{'id': r['id'], 'parsed_events': [{'type': e['type'], 'trigger': e['trigger'],
                        'args': [{'role': a['role'], 'text': a['text']} for a in e['args']]}
                        for e in canonical_events({'document': r['tokens'], 'events': r['gold_events']})],
                        'invalid': False, 'parse_diagnostics': {}, 'hit_max_new_tokens': False}
                       for r in data_rows]
    ids = [p['id'] for p in predictions]
    if len(set(ids)) != len(ids) or not set(ids).issubset(by_id):
        raise ValueError('Duplicate or unknown prediction IDs')
    golds, preds, output_docs, per_doc = [], [], [], []
    totals = Counter()
    per_type = {t: Counter() for t in EVENT_TYPES}
    for pred in predictions:
        row = by_id[pred['id']]
        if 'aligned_events' in pred:
            events = pred['aligned_events']
            stats = pred.get('alignment', {'events': len(events), 'aligned_events': len(events),
                                           'unaligned_triggers': 0, 'arguments': 0,
                                           'aligned_arguments': 0, 'unaligned_arguments': 0,
                                           'trigger_methods': {}, 'argument_methods': {}})
        else:
            events, stats = align_forest(row['tokens'], row['sentences'], pred['parsed_events'])
        events = strip_private(events)
        golds.append(row['gold_events'])
        preds.append(events)
        output_docs.append({'doc_id': row['id'], 'events': events})
        totals.update({k: v for k, v in stats.items() if isinstance(v, int)})
        totals['invalid_documents'] += bool(pred.get('invalid'))
        totals['hit_max_new_tokens'] += bool(pred.get('hit_max_new_tokens'))
        for key, value in pred.get('parse_diagnostics', {}).items():
            if key != 'invalid':
                totals['parse_' + key] += value
        for method, n in stats['trigger_methods'].items():
            totals['trigger_method_' + method] += n
        for method, n in stats['argument_methods'].items():
            totals['argument_method_' + method] += n
        gold_types = Counter(e['event_type'] for e in row['gold_events'])
        pred_types = Counter(e['event_type'] for e in events)
        for t in EVENT_TYPES:
            per_type[t]['gold'] += gold_types[t]
            per_type[t]['predicted'] += pred_types[t]
        doc_scores = official_scores([row['gold_events']], [events])
        per_doc.append({'id': row['id'], 'gold_events': len(row['gold_events']), 'predicted_events': len(events),
                        'invalid': bool(pred.get('invalid')), 'TC_f1': doc_scores['TC_f1'], 'AC_f1': doc_scores['AC_f1'],
                        'raw_output': pred.get('raw_output')})
    metrics = {'documents': len(predictions), 'scope': 'subset' if len(predictions) != len(data_rows) else 'full',
               'official': official_scores(golds, preds), 'alignment': dict(totals),
               'per_event_type': {t: dict(c) for t, c in per_type.items()},
               'gold_events': sum(len(g) for g in golds), 'predicted_events': sum(len(p) for p in preds)}
    return metrics, output_docs, per_doc


def run_complex_eval(prediction_json, python):
    """Official complex-scenario subsets (test only). Returns parsed 'Name: METRIC F1=xx.xx' lines."""
    result = subprocess.run([python, 'eval_complex_quick_f1.py', '--outputs', str(prediction_json), '--id_check'],
                            cwd=PROJECT_ROOT, capture_output=True, text=True)
    parsed = {}
    for line in result.stdout.splitlines():
        match = re.match(r'(\w+): (\w+) F1=([\d.]+)', line.strip())
        if match:
            parsed[f'{match.group(1)}_{match.group(2)}_f1'] = float(match.group(3)) / 100
    return {'exit_code': result.returncode, 'scores': parsed, 'stdout': result.stdout[-4000:],
            'stderr': result.stderr[-2000:]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--predictions', type=Path)
    parser.add_argument('--output', type=Path, required=True, help='metrics JSON; predictions/per-doc written beside it')
    parser.add_argument('--oracle', action='store_true', help='score the canonical gold tree through the aligner')
    parser.add_argument('--allow-subset', action='store_true')
    parser.add_argument('--complex', action='store_true', help='also run the official complex-scenario script (test only)')
    args = parser.parse_args()
    data = read_jsonl(args.data)
    if args.oracle:
        predictions = None
    else:
        if not args.predictions:
            parser.error('--predictions or --oracle is required')
        predictions = read_jsonl(args.predictions)
        if not args.allow_subset and len(predictions) != len(data):
            raise ValueError('Missing predictions; use --allow-subset only for declared smoke runs')
    metrics, output_docs, per_doc = evaluate(data, predictions, oracle=args.oracle)
    metrics.update(data=str(args.data.resolve()), predictions=str(args.predictions.resolve()) if args.predictions else None,
                   oracle=args.oracle)
    output = args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    pred_json = output.with_suffix('.scievents.json')
    with pred_json.open('x', encoding='utf-8') as stream:
        json.dump(output_docs, stream, ensure_ascii=False)
    if args.complex and metrics['scope'] == 'full' and data[0]['split'] == 'test':
        metrics['complex'] = run_complex_eval(pred_json, sys.executable)
    with output.open('x', encoding='utf-8') as stream:
        json.dump(metrics, stream, ensure_ascii=False, indent=2)
    with output.with_suffix('.docs.jsonl').open('x', encoding='utf-8') as stream:
        for row in per_doc:
            stream.write(json.dumps(row, ensure_ascii=False) + '\n')
    print(json.dumps({k: metrics[k] for k in ('documents', 'scope', 'official', 'alignment')}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
