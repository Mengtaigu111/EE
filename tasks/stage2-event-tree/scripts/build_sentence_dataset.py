#!/usr/bin/env python3
"""Build Experiment C: one full-context, target-sentence row per document sentence."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

from align_events import sentence_bounds
from linearize import canonical_events, linearize, parse_forest
from prompts import make_sentence_prompt

PROJECT_ROOT = Path(__file__).resolve().parents[3]


def _event_in_sentence(event, lo, hi):
    offsets = event['trigger_offsets']
    return bool(offsets) and all(lo <= offset < hi for offset in offsets)


def build_rows_for_document(doc, split):
    tokens = doc['document']
    if ' '.join(doc['sentences']).split() != tokens:
        raise ValueError(f'{doc["doc_id"]}: sentences do not tile document tokens')
    bounds = sentence_bounds(doc['sentences'])
    events = canonical_events(doc)
    rows = []
    sentence_event_counts = []
    cross_sentence_arguments = 0
    cross_sentence_events = 0
    for sentence_index, ((lo, hi), sentence) in enumerate(zip(bounds, doc['sentences'])):
        target_events = [event for event in events if _event_in_sentence(event, lo, hi)]
        for event in target_events:
            cross_arguments = sum(not all(lo <= offset < hi for offset in arg['offsets'])
                                  for arg in event['args'])
            cross_sentence_arguments += cross_arguments
            cross_sentence_events += bool(cross_arguments)
        completion = linearize(target_events)
        parsed, diagnostics = parse_forest(completion)
        if diagnostics['invalid'] or diagnostics['dropped_events'] or diagnostics['dropped_args']:
            raise ValueError(f'{doc["doc_id"]} sentence {sentence_index}: tree did not roundtrip: {diagnostics}')
        expected = [(event['type'], event['trigger'], [(arg['role'], arg['text']) for arg in event['args']])
                    for event in target_events]
        actual = [(event['type'], event['trigger'], [(arg['role'], arg['text']) for arg in event['args']])
                  for event in parsed]
        if actual != expected:
            raise ValueError(f'{doc["doc_id"]} sentence {sentence_index}: parser/linearizer mismatch')
        target_gold_events = [raw for raw in doc['events']
                             if _event_in_sentence({'trigger_offsets': raw['trigger']['offsets']}, lo, hi)]
        rows.append({
            'id': f'{doc["doc_id"]}::sentence-{sentence_index}',
            'document_id': doc['doc_id'],
            'split': split,
            'mode': 'sentence',
            'sentence_index': sentence_index,
            'target_bounds': [lo, hi],
            'target_sentence': sentence,
            'tokens': tokens,
            'sentences': doc['sentences'],
            'abstract': ' '.join(tokens),
            'prompt': make_sentence_prompt(' '.join(tokens), doc['sentences'], sentence_index),
            'completion': completion,
            'target_events': target_gold_events,
            'n_events': len(target_events),
        })
        sentence_event_counts.append(len(target_events))
    return rows, {
        'documents': 1,
        'sentences': len(rows),
        'event_bearing_sentences': sum(n > 0 for n in sentence_event_counts),
        'no_event_sentences': sum(n == 0 for n in sentence_event_counts),
        'events': len(events),
        'max_events_per_sentence': max(sentence_event_counts, default=0),
        'cross_sentence_arguments': cross_sentence_arguments,
        'cross_sentence_events': cross_sentence_events,
    }


def build_split(project_root, split, output_dir):
    source = project_root / f'data/SciEvents/{split}.json'
    docs = json.loads(source.read_text(encoding='utf-8'))
    all_rows, totals = [], Counter()
    max_events_per_sentence = 0
    for doc in docs:
        rows, stats = build_rows_for_document(doc, split)
        all_rows.extend(rows)
        max_events_per_sentence = max(max_events_per_sentence, stats['max_events_per_sentence'])
        totals.update({key: value for key, value in stats.items() if key != 'max_events_per_sentence'})
    totals['max_events_per_sentence'] = max_events_per_sentence
    output = output_dir / f'{split}.jsonl'
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('w', encoding='utf-8') as stream:
        for row in all_rows:
            stream.write(json.dumps(row, ensure_ascii=False, separators=(',', ':')) + '\n')
    return output, dict(totals)


def build_all(project_root, output_dir):
    manifest = {'created_at': datetime.now(timezone.utc).isoformat(), 'mode': 'sentence',
                'rules': {'context': 'full abstract', 'trigger_scope': 'target sentence',
                          'argument_scope': 'full abstract', 'hints': 'none',
                          'empty_target': '(ROOT)'}, 'sources': {}, 'splits': {}}
    for split in ('train', 'dev', 'test'):
        source = project_root / f'data/SciEvents/{split}.json'
        manifest['sources'][str(source.relative_to(project_root))] = hashlib.sha256(source.read_bytes()).hexdigest()
        output, stats = build_split(project_root, split, output_dir)
        manifest['splits'][split] = stats
        with output.open(encoding='utf-8') as stream:
            manifest['splits'][split]['rows'] = sum(1 for _ in stream)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest['outputs'] = {str(path.relative_to(output_dir)): hashlib.sha256(path.read_bytes()).hexdigest()
                           for path in sorted(output_dir.glob('*.jsonl'))}
    (output_dir / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n')
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project-root', type=Path, default=PROJECT_ROOT)
    parser.add_argument('--output-dir', type=Path,
                        default=Path(__file__).resolve().parents[1] / 'data/sentence-c')
    args = parser.parse_args()
    print(json.dumps(build_all(args.project_root, args.output_dir), ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
