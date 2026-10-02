#!/usr/bin/env python3
"""Derive byte-preserving cumulative splits from the frozen AppendixNoCopy run."""
import argparse
from collections import Counter
import csv
import io
import hashlib
import json
import math
from pathlib import Path
import random
from statistics import median

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[2]
DEFAULT_BASELINE = HERE.parent / 'runs/C-AppendixNoCopy-42-20261002/run.json'
DEFAULT_ASSIGNMENT = HERE.parent / 'difficulty-analysis/mean_percentile_difficulty_assignments.csv'
TAU = 0.4275338928340865


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def unique(items, key, label):
    indexed = {}
    for item in items:
        identifier = item[key]
        if identifier in indexed:
            raise ValueError(f'duplicate {label} ID: {identifier}')
        indexed[identifier] = item
    return indexed


def validate_assignments(rows, assignments, tau):
    data = unique(rows, 'id', 'data')
    labels = unique(assignments, 'sample_id', 'assignment')
    if data.keys() != labels.keys():
        raise ValueError('assignment IDs do not cover exactly the original training set')
    buckets = {'B1': [], 'B2': [], 'ROOT': []}
    for row in rows:
        identifier = row['id']
        label = labels[identifier]
        n = row['n_events']
        if 'target_events' in row and n != len(row['target_events']):
            raise ValueError(f'gold event count mismatch: {identifier}')
        if n < 0 or (n == 0) != (row['completion'] == '(ROOT)'):
            raise ValueError(f'ROOT completion/gold count mismatch: {identifier}')
        score = label['score_mean']
        if n == 0:
            if label['bucket'] != 'ROOT' or score != 'NA':
                raise ValueError(f'ROOT has difficulty/bucket: {identifier}')
            bucket = 'ROOT'
        else:
            score = float(score)
            if not math.isfinite(score) or not 0 <= score <= 1:
                raise ValueError(f'invalid difficulty: {identifier}')
            bucket = 'B1' if score <= tau else 'B2'
            if label['bucket'] != bucket:
                raise ValueError(f'fixed threshold/tie policy mismatch: {identifier}')
        buckets[bucket].append(identifier)
    return buckets


def save_bytes(path, content):
    if path.exists():
        if path.read_bytes() != content:
            raise FileExistsError(f'refusing to overwrite different derived file: {path}')
        return
    with path.open('xb') as stream:
        stream.write(content)


def save_json(path, value):
    save_bytes(path, (json.dumps(value, indent=2, ensure_ascii=False)+'\n').encode('utf-8'))


def save_ids(path, ids):
    save_bytes(path, ''.join(identifier+'\n' for identifier in ids).encode('utf-8'))


def build(output_dir=HERE, baseline_path=DEFAULT_BASELINE, assignment_path=DEFAULT_ASSIGNMENT):
    output_dir = Path(output_dir).resolve()
    baseline_path, assignment_path = Path(baseline_path).resolve(), Path(assignment_path).resolve()
    baseline = json.loads(baseline_path.read_text())
    if baseline['status'] != 'completed':
        raise ValueError('baseline must be completed')
    data_dir = Path(baseline['data_dir'])
    train_path = data_dir / 'train.jsonl'
    lines = train_path.read_bytes().splitlines(keepends=True)
    rows = [json.loads(line) for line in lines]
    if sha(train_path) != baseline['audit']['data_sha256']:
        raise ValueError('baseline train hash mismatch')
    if any(row['mode'] != 'sentence-appendix-schema' for row in rows):
        raise ValueError('expected frozen improved AppendixNoCopy prompt')
    with assignment_path.open(newline='') as stream:
        assignments = [row for row in csv.DictReader(stream)
                       if float(row['fraction']) == .5 and row['tie_policy'] == 'all_ties_easy']
    buckets = validate_assignments(rows, assignments, TAU)
    if [len(buckets[k]) for k in ('B1','B2','ROOT')] != [5427,5277,2124]:
        raise ValueError('fixed bucket counts differ from approved assignment')
    root_count = round(len(buckets['B1'])*len(buckets['ROOT'])/(len(buckets['B1'])+len(buckets['B2'])))
    root_subset = random.Random(42).sample(buckets['ROOT'], root_count)
    chosen = set(buckets['B1']) | set(root_subset)
    stage1 = [row['id'] for row in rows if row['id'] in chosen]
    stage2 = [row['id'] for row in rows]
    training_dir = baseline_path.parent / 'training'
    trainer = json.loads((training_dir/'trainer_state.json').read_text())
    metrics = json.loads((training_dir/'train_metrics.json').read_text())
    config = json.loads((training_dir/'config.json').read_text())
    total_steps = trainer['global_step']
    if total_steps != metrics['global_step'] or total_steps != 4010:
        raise ValueError('baseline optimizer budget mismatch')
    output_dir.mkdir(parents=True, exist_ok=True)
    stage1_path = output_dir/'mean_curriculum_stage1.jsonl'
    stage2_path = output_dir/'mean_curriculum_stage2.jsonl'
    save_bytes(stage1_path, b''.join(line for row,line in zip(rows,lines) if row['id'] in chosen))
    save_bytes(stage2_path, train_path.read_bytes())
    for name,ids in [('stage1',stage1),('stage2',stage2),('root_stage1',root_subset)]:
        save_ids(output_dir/f'mean_curriculum_{name}_ids.txt',ids)
    manifest = {
        'total_train':len(rows),'positive_train':len(buckets['B1'])+len(buckets['B2']),
        'root_total':len(buckets['ROOT']),'B1_count':len(buckets['B1']),'B2_count':len(buckets['B2']),
        'difficulty_formula':'0.5 * ((1 - 1/N_e(s)) + (1 - P_mean(s)))',
        'difficulty_threshold':TAU,'fraction':.5,'tie_policy':'all_ties_easy',
        'root_stage1_count':root_count,'root_stage1_seed':42,'root_sampling':'random.Random(42).sample in original ROOT order',
        'stage1_total_count':len(stage1),'stage2_total_count':len(stage2),
        'full_train_root_ratio':len(buckets['ROOT'])/len(rows),
        'stage1_root_ratio':root_count/len(stage1),
        'stage2_root_ratio':len(buckets['ROOT'])/len(stage2),
        'baseline_total_optimizer_steps':total_steps,'stage1_optimizer_steps':total_steps//2,
        'stage2_optimizer_steps':total_steps-total_steps//2,'base_model':config['model_path'],
        'training_seed':42,'source_train_file':str(train_path),'difficulty_assignment_file':str(assignment_path),
        'baseline_run_file':str(baseline_path),'baseline_training_config':config,
        'stage1_data':str(stage1_path),'stage2_data':str(stage2_path),
        'dev_data':str(data_dir/'dev.jsonl'),'test_data':str(data_dir/'test.jsonl'),
        'document_data_dir':baseline['document_data_dir'],'root_stage1_ids':root_subset,
        'source_hashes':{str(p):sha(p) for p in [train_path,data_dir/'dev.jsonl',data_dir/'test.jsonl',assignment_path,baseline_path,training_dir/'config.json',training_dir/'trainer_state.json',training_dir/'train_metrics.json']},
        'baseline_dev_checkpoints':baseline['dev_checkpoints'],
        'selection_metric':'official.AC_f1; tie: later global step',
    }
    if abs(manifest['stage1_root_ratio']-manifest['full_train_root_ratio'])*100 >= .2:
        raise ValueError('Stage1 ROOT proportion differs by >= 0.2 percentage points')
    for item in baseline['dev_checkpoints']:
        manifest['source_hashes'][item['metrics']] = sha(item['metrics'])
    positive = [row for row in rows if row['n_events']]
    event_scores = {row['id']:1-1/row['n_events'] for row in positive}
    event_tau = median(event_scores.values())
    random_easy = set(random.Random(42).sample([row['id'] for row in positive],len(buckets['B1'])))
    for name,easy,threshold in [
            ('event_count',{k for k,v in event_scores.items() if v <= event_tau},event_tau),
            ('random_half',random_easy,None)]:
        control_labels = []
        for row in rows:
            bucket = 'ROOT' if not row['n_events'] else ('B1' if row['id'] in easy else 'B2')
            control_labels.append({'sample_id':row['id'],'bucket':bucket,
                                   'stage1_included':row['id'] in easy or row['id'] in set(root_subset)})
        stream = io.StringIO(newline='')
        writer = csv.DictWriter(stream,fieldnames=['sample_id','bucket','stage1_included'])
        writer.writeheader()
        writer.writerows(control_labels)
        save_bytes(output_dir/f'{name}_control_assignments.csv',stream.getvalue().encode('utf-8'))
        save_json(output_dir/f'{name}_control_config.json',{
            'auto_train':False,'seed':42,'stage1_positive_count':len(easy),
            'stage1_total_count':len(easy)+root_count,'root_stage1_ids':root_subset,
            'stage2_total_count':len(rows),'threshold':threshold,'tie_policy':'all_ties_easy' if threshold is not None else None,
            'difficulty_formula':'1 - 1/N_e(s)' if threshold is not None else 'random positive subset',
            'baseline_total_optimizer_steps':total_steps,'stage1_optimizer_steps':total_steps//2,
            'stage2_optimizer_steps':total_steps-total_steps//2,'baseline_run_file':str(baseline_path),
            'source_train_file':str(train_path),'root_subset_shared_with_main':True})
    # Independent disk readback proves no row was reconstructed or silently lost.
    read1 = stage1_path.read_bytes().splitlines(keepends=True)
    read2 = stage2_path.read_bytes()
    originals = {row['id']:line for row,line in zip(rows,lines)}
    ids1 = [json.loads(line)['id'] for line in read1]
    if ids1 != stage1 or len(set(ids1)) != len(ids1) or set(ids1) != chosen:
        raise ValueError('Stage1 readback membership failed')
    if any(line != originals[json.loads(line)['id']] for line in read1) or read2 != train_path.read_bytes():
        raise ValueError('byte preservation readback failed')
    manifest['output_hashes'] = {str(p):sha(p) for p in sorted(output_dir.glob('*'))
                                 if p.is_file() and (p.name.startswith('mean_curriculum_') or '_control_' in p.name)
                                 and p.name != 'mean_curriculum_manifest.json'}
    manifest['verification'] = {'byte_preserving_stage1':True,'byte_identical_stage2':True,
                                'unique_complete_assignment_join':True,'fixed_threshold_and_gold_counts':True}
    save_json(output_dir/'mean_curriculum_manifest.json',manifest)
    return manifest


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',type=Path,default=HERE)
    parser.add_argument('--baseline-run',type=Path,default=DEFAULT_BASELINE)
    parser.add_argument('--assignments',type=Path,default=DEFAULT_ASSIGNMENT)
    args = parser.parse_args()
    result = build(args.output_dir,args.baseline_run,args.assignments)
    print(json.dumps({k:result[k] for k in ['stage1_total_count','stage2_total_count','root_stage1_count','baseline_total_optimizer_steps','stage1_optimizer_steps','stage2_optimizer_steps']},indent=2))
