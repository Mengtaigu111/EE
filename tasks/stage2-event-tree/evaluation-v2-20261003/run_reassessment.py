#!/usr/bin/env python3
"""CPU-only v2 reassessment from immutable sentence raw_output files."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
SCRIPTS = HERE / 'scripts'
RUNS = ROOT / 'tasks/stage2-event-tree/runs'
DATA = ROOT / 'tasks/stage2-event-tree/data'
GROUPS = {
    'curriculum': ('C-CL-Mean-UserV2-Easy2-42-20261002', 'sentence-c-user-v2-20261002',
                   (802, 814, 1604, 2406, 3208, 4010)),
    'baseline': ('C-AppendixNoCopy-42-20261002', 'sentence-c-appendix-no-copy-20261002',
                 (802, 1604, 2406, 3208, 4010)),
}
COMPLEX_KEYS = {'Consecutive_AC_f1', 'Overlap_TC_f1', 'Overlap_AC_f1',
                'Reverse_AC_f1', 'Subevent_TC_f1', 'Subevent_AC_f1', 'Subevent_EC_f1'}


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def read_jsonl(path):
    with path.open(encoding='utf-8') as stream:
        return [json.loads(line) for line in stream if line.strip()]


def atomic_json(path, data):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    temporary.replace(path)


def verify_raw(path, sentence_data, checkpoint):
    marker = Path(str(path) + '.complete.json')
    config_path = Path(str(path) + '.config.json')
    if not (path.is_file() and marker.is_file() and config_path.is_file()):
        raise FileNotFoundError(f'incomplete raw prediction: {path}')
    complete = json.loads(marker.read_text())
    config = json.loads(config_path.read_text())
    source_hash = sha(sentence_data)
    output_hash = sha(path)
    rows = read_jsonl(sentence_data)
    preds = read_jsonl(path)
    row_ids = [r['id'] for r in rows]
    pred_ids = [r['id'] for r in preds]
    if (complete.get('status') != 'completed' or complete.get('count') != len(rows)
            or complete.get('output_sha256') != output_hash
            or config.get('data_sha256') != source_hash
            or Path(config.get('data_path', '')).resolve() != sentence_data.resolve()
            or Path(config.get('adapter', '')).resolve() != checkpoint.resolve()
            or config.get('adapter_sha256') != sha(checkpoint / 'adapter_model.safetensors')
            or len(pred_ids) != len(set(pred_ids)) or pred_ids != row_ids
            or any(not isinstance(p.get('raw_output'), str) for p in preds)):
        raise ValueError(f'raw prediction provenance/ID coverage failed: {path}')
    return {'raw_path': str(path), 'raw_sha256': output_hash, 'complete_marker_sha256': sha(marker),
            'config_sha256': sha(config_path), 'sentence_data_sha256': source_hash,
            'sentence_count': len(rows), 'id_coverage': 'exact_order_and_unique',
            'checkpoint': str(checkpoint), 'adapter_sha256': config['adapter_sha256']}


def checkpoint_path(run_name, step):
    run_dir = RUNS / run_name
    if run_name.startswith('C-CL-'):
        phase = 'stage1' if step <= 814 else 'stage2'
    else:
        phase = 'training'
    return run_dir / phase / 'checkpoints' / f'checkpoint-{step}'


def reassess(group, split, step):
    run_name, data_name, _ = GROUPS[group]
    checkpoint = checkpoint_path(run_name, step)
    sentence_data = DATA / data_name / f'{split}.jsonl'
    document_data = DATA / 'none' / f'{split}.jsonl'
    raw = RUNS / run_name / 'checkpoint-eval' / f'{split}-checkpoint-{step}.predictions.jsonl'
    evidence = verify_raw(raw, sentence_data, checkpoint)
    outdir = HERE / 'results' / group
    outdir.mkdir(parents=True, exist_ok=True)
    prefix = outdir / f'{split}-checkpoint-{step}'
    merged = prefix.with_suffix('.merged.jsonl')
    metrics_path = prefix.with_suffix('.metrics.json')
    record_path = prefix.with_suffix('.provenance.json')
    if record_path.exists():
        record = json.loads(record_path.read_text())
        metrics = json.loads(metrics_path.read_text())
        current_code = {p.name: sha(p) for p in sorted(SCRIPTS.glob('*.py'))}
        if (record['input'] != evidence or record['metrics_sha256'] != sha(metrics_path)
                or record['merged_sha256'] != sha(merged)
                or record['document_data_sha256'] != sha(document_data)
                or record['code_sha256'] != current_code
                or metrics['scope'] != 'full' or metrics['official'] != record['official']
                or (split == 'test' and (metrics.get('complex', {}).get('exit_code') != 0
                                          or set(metrics['complex']['scores']) != COMPLEX_KEYS))):
            raise ValueError(f'existing v2 result disagrees with inputs: {record_path}')
        return record['official']['AC_f1']
    if merged.exists() or metrics_path.exists():
        raise FileExistsError(f'partial v2 output: {prefix}')
    env = {**os.environ, 'CUDA_VISIBLE_DEVICES': ''}
    subprocess.run([sys.executable, str(SCRIPTS / 'merge_sentence_predictions.py'),
                    '--data', str(sentence_data), '--predictions', str(raw), '--output', str(merged)],
                   cwd=ROOT, env=env, check=True)
    docs = read_jsonl(document_data)
    merged_rows = read_jsonl(merged)
    if len(merged_rows) != len(docs) or {x['id'] for x in merged_rows} != {x['id'] for x in docs}:
        raise ValueError(f'merged document ID coverage failed: {merged}')
    subprocess.run([sys.executable, str(SCRIPTS / 'evaluate.py'), '--data', str(document_data),
                    '--predictions', str(merged), '--output', str(metrics_path)]
                   + (['--complex'] if split == 'test' else []), cwd=ROOT, env=env, check=True)
    metrics = json.loads(metrics_path.read_text())
    if metrics['scope'] != 'full' or metrics['documents'] != len(docs):
        raise ValueError(f'incomplete metrics: {metrics_path}')
    if split == 'test' and (metrics.get('complex', {}).get('exit_code') != 0
                            or set(metrics['complex']['scores']) != COMPLEX_KEYS):
        raise ValueError(f'incomplete complex metrics: {metrics_path}')
    record = {'input': evidence, 'document_data': str(document_data),
              'document_data_sha256': sha(document_data), 'merged_sha256': sha(merged),
              'metrics_sha256': sha(metrics_path), 'official': metrics['official'],
              'complex': metrics.get('complex'), 'code_sha256': {
                  p.name: sha(p) for p in sorted(SCRIPTS.glob('*.py'))}}
    atomic_json(record_path, record)
    return metrics['official']['AC_f1']


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--split', choices=('dev', 'test'), required=True)
    parser.add_argument('--group', choices=tuple(GROUPS), help='run one group only')
    args = parser.parse_args()
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise RuntimeError("Run with CUDA_VISIBLE_DEVICES='' for CPU reassessment")
    if args.split == 'dev':
        summary = {}
        for group in ((args.group,) if args.group else GROUPS):
            run_name, _, steps = GROUPS[group]
            values = {str(step): reassess(group, 'dev', step) for step in steps}
            best = max(steps, key=lambda step: (values[str(step)], step))
            old = json.loads((RUNS / run_name / 'run.json').read_text())
            summary[group] = {'scores': values, 'best_step': best,
                              'old_best_step': int(Path(old['best_checkpoint']).name.split('-')[-1])
                              if old.get('best_checkpoint') else None,
                              'old_scores': {str(x['step']): x['AC_f1']
                                             for x in old.get('dev_checkpoints', [])}}
        if args.group:
            raise ValueError('dev selection must include both groups')
        atomic_json(HERE / 'results/dev_selection.json', summary)
        print(json.dumps(summary, ensure_ascii=False, indent=2))
    else:
        selection = json.loads((HERE / 'results/dev_selection.json').read_text())
        groups = (args.group,) if args.group else GROUPS
        result = {group: {'step': selection[group]['best_step'],
                          'AC_f1': reassess(group, 'test', selection[group]['best_step'])}
                  for group in groups}
        if not args.group:
            atomic_json(HERE / 'results/test_selection.json', result)
        print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
