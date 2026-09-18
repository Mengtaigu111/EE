#!/usr/bin/env python3
"""Build Stage-2 document-level datasets: abstract -> canonical event forest, with optional hints.

Modes: none | flat | typed. Train hints are gold hints noised with a profile calibrated
on Stage-1 dev errors (never Stage-1 train predictions, which are overfit). Dev/test hints
come from a Stage-1 run; an extra '<mode>-oracle' variant uses gold hints for the upper bound.
"""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

from hints import gold_hints, load_stage1_hints, noise_profile_from_stage1, noised_hints, summarize
from linearize import canonical_events, linearize, linearize_substructures, parse_forest
from prompts import HINT_MODES, MODES, NATURAL_MODES, make_prompt

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_STAGE1_RUN = PROJECT_ROOT / 'tasks/stage1-typed-mentions/runs/E1-42-20260916-204747'


def build_row(doc, split, mode, hints):
    tokens = doc['document']
    if ' '.join(doc['sentences']).split() != tokens:
        raise ValueError(f'{doc["doc_id"]}: sentences do not tile the document tokens')
    events = canonical_events(doc)
    natural = mode in NATURAL_MODES
    if mode == 'sub':
        completion = linearize_substructures(events, natural)
    else:
        completion = linearize(events, natural)
        parsed, diag = parse_forest(completion)
        if diag['invalid'] or diag['dropped_events'] or diag['dropped_args']:
            raise ValueError(f'{doc["doc_id"]}: canonical tree does not roundtrip: {diag}')
        if [(e['type'], e['trigger'], [(a['role'], a['text']) for a in e['args']]) for e in parsed] != \
           [(e['type'], e['trigger'], [(a['role'], a['text']) for a in e['args']]) for e in events]:
            raise ValueError(f'{doc["doc_id"]}: parser/linearizer mismatch')
    abstract = ' '.join(tokens)
    return {'id': doc['doc_id'], 'document_id': doc['doc_id'], 'split': split, 'mode': mode,
            'tokens': tokens, 'sentences': doc['sentences'], 'abstract': abstract,
            'hints': [{'text': h['text'], 'type': h['type']} for h in hints],
            'prompt': make_prompt(abstract, mode, hints), 'completion': completion,
            'gold_events': doc['events'], 'n_events': len(events)}


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8') as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, separators=(',', ':')) + '\n')


def build_all(project_root, output_dir, stage1_run, seed):
    manifest = {'created_at': datetime.now(timezone.utc).isoformat(), 'seed': seed, 'sources': {}, 'splits': {},
                'stage1_run': str(stage1_run),
                'rules': {'tree': 'ROOT -> (EventType (Trigger text) (Role text)*)*; events by trigger offset then type; '
                                  'roles in ontology order; same-role args by offset; text = tokens in sorted offset order',
                          'train_hints': 'gold hints with calibrated noise (per-type miss, type flip, boundary jitter, spurious spans)',
                          'eval_hints': 'Stage-1 predictions aligned with the Stage-1 aligner, deduplicated by (offsets,type)',
                          'oracle_hints': 'gold hints, same prompt as the noised/predicted variants',
                          'natural': 'event-type codes replaced by natural-language names in schema and tree (Text2Event w/o ES ablation)',
                          'noschema': 'no EventType->Roles schema in the prompt',
                          'sub': 'Text2Event curriculum stage: flat (label "span") pairs for triggers and arguments'}}
    profile = noise_profile_from_stage1(stage1_run / 'dev.metrics.json')
    manifest['noise_profile'] = profile
    stage1 = {split: load_stage1_hints(stage1_run / f'{split}.predictions.jsonl') for split in ('dev', 'test')}
    for split in ('train', 'dev', 'test'):
        raw_path = project_root / f'data/SciEvents/{split}.json'
        manifest['sources'][str(raw_path.relative_to(project_root))] = hashlib.sha256(raw_path.read_bytes()).hexdigest()
        docs = json.loads(raw_path.read_text(encoding='utf-8'))
        if len({d['doc_id'] for d in docs}) != len(docs):
            raise ValueError(f'Duplicate doc_id in {split}')
        stats = Counter(documents=len(docs), events=sum(len(d['events']) for d in docs))
        golds = {d['doc_id']: gold_hints(d) for d in docs}
        if split == 'train':
            noised = {d['doc_id']: noised_hints(d, golds[d['doc_id']], profile, seed) for d in docs}
            eval_hints = None
        else:
            noised = None
            eval_hints = stage1[split]
            missing = [d['doc_id'] for d in docs if d['doc_id'] not in eval_hints]
            if missing:
                raise ValueError(f'{split}: {len(missing)} documents without Stage-1 predictions, e.g. {missing[:3]}')
        for mode in MODES:
            if split == 'train':
                rows = [build_row(d, split, mode, noised[d['doc_id']] if mode in HINT_MODES else []) for d in docs]
                write_jsonl(output_dir / mode / f'{split}.jsonl', rows)
            else:
                rows = [build_row(d, split, mode, eval_hints[d['doc_id']] if mode in HINT_MODES else []) for d in docs]
                write_jsonl(output_dir / mode / f'{split}.jsonl', rows)
                if mode in HINT_MODES:
                    oracle = [build_row(d, split, mode, golds[d['doc_id']]) for d in docs]
                    write_jsonl(output_dir / f'{mode}-oracle' / f'{split}.jsonl', oracle)
        stats['gold_hints'] = sum(len(v) for v in golds.values())
        if noised is not None:
            stats.update({'noised_' + k: v for k, v in summarize(noised).items() if k != 'by_type'})
        if eval_hints is not None:
            stats.update({'stage1_' + k: v for k, v in summarize(eval_hints).items() if k != 'by_type'})
        manifest['splits'][split] = dict(stats)
    manifest['outputs'] = {str(p.relative_to(output_dir)): hashlib.sha256(p.read_bytes()).hexdigest()
                           for p in sorted(output_dir.rglob('*.jsonl'))}
    (output_dir / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n')
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project-root', type=Path, default=PROJECT_ROOT)
    parser.add_argument('--output-dir', type=Path, default=Path(__file__).resolve().parents[1] / 'data')
    parser.add_argument('--stage1-run', type=Path, default=DEFAULT_STAGE1_RUN)
    parser.add_argument('--seed', type=int, default=42, help='noise seed for train hints')
    args = parser.parse_args()
    manifest = build_all(args.project_root, args.output_dir, args.stage1_run, args.seed)
    print(json.dumps({k: v for k, v in manifest.items() if k != 'outputs'}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
