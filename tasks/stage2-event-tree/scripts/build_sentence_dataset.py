#!/usr/bin/env python3
"""Build Experiment C: one full-context, target-sentence row per document sentence."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

from align_events import sentence_bounds
from linearize import canonical_events, linearize, linearize_semantic, parse_forest
from prompts import make_appendix_sentence_prompt, make_semantic_sentence_prompt, make_sentence_prompt

PROJECT_ROOT = Path(__file__).resolve().parents[3]


def _event_in_sentence(event, lo, hi):
    offsets = event['trigger_offsets']
    return bool(offsets) and all(lo <= offset < hi for offset in offsets)


def _original_gold_signature(tokens, event):
    """Independent gold comparison; never read canonicalized text or offsets."""
    def text(mention):
        if 'tokens' in mention:
            return ' '.join(mention['tokens'])
        if 'text' in mention:
            return mention['text']
        return ' '.join(tokens[i] for i in mention['offsets'])
    return (event['event_type'], text(event['trigger']),
            tuple(sorted((arg['argument_type'], text(arg)) for arg in event['arguments'])))


def _mode_name(semantic_path=False, appendix_schema=False):
    if semantic_path and appendix_schema:
        raise ValueError('semantic_path and appendix_schema are mutually exclusive')
    if semantic_path:
        return 'sentence-semantic-path'
    if appendix_schema:
        return 'sentence-appendix-schema'
    return 'sentence'


def _prompt_builder(semantic_path=False, appendix_schema=False):
    if semantic_path and appendix_schema:
        raise ValueError('semantic_path and appendix_schema are mutually exclusive')
    if semantic_path:
        return make_semantic_sentence_prompt
    if appendix_schema:
        return make_appendix_sentence_prompt
    return make_sentence_prompt


def build_rows_for_document(doc, split, semantic_path=False, appendix_schema=False):
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
        completion = linearize_semantic(target_events) if semantic_path else linearize(target_events)
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
        gold_signatures = Counter(_original_gold_signature(tokens, raw) for raw in target_gold_events)
        parsed_signatures = Counter((event['type'], event['trigger'],
            tuple(sorted((arg['role'], arg['text']) for arg in event['args']))) for event in parsed)
        if parsed_signatures != gold_signatures:
            raise ValueError(f'{doc["doc_id"]} sentence {sentence_index}: tree differs from original gold')
        rows.append({
            'id': f'{doc["doc_id"]}::sentence-{sentence_index}',
            'document_id': doc['doc_id'],
            'split': split,
            'mode': _mode_name(semantic_path=semantic_path, appendix_schema=appendix_schema),
            'sentence_index': sentence_index,
            'target_bounds': [lo, hi],
            'target_sentence': sentence,
            'tokens': tokens,
            'sentences': doc['sentences'],
            'abstract': ' '.join(tokens),
            'prompt': _prompt_builder(semantic_path=semantic_path, appendix_schema=appendix_schema)(
                ' '.join(tokens), doc['sentences'], sentence_index),
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


def build_split(project_root, split, output_dir, semantic_path=False, appendix_schema=False):
    source = project_root / f'data/SciEvents/{split}.json'
    docs = json.loads(source.read_text(encoding='utf-8'))
    all_rows, totals = [], Counter()
    max_events_per_sentence = 0
    for doc in docs:
        rows, stats = build_rows_for_document(
            doc, split, semantic_path=semantic_path, appendix_schema=appendix_schema)
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


def build_all(project_root, output_dir, semantic_path=False, appendix_schema=False):
    manifest = {'created_at': datetime.now(timezone.utc).isoformat(),
                'mode': _mode_name(semantic_path=semantic_path, appendix_schema=appendix_schema),
                'rules': {'context': 'full abstract', 'trigger_scope': 'target sentence',
                          'argument_scope': 'full abstract', 'hints': 'none',
                          'tree': 'semantic path' if semantic_path else 'event type root',
                          'schema': 'Appendix F.2 exact' if appendix_schema else (
                              'semantic prototypes' if semantic_path else 'ontology.py gloss'),
                          'empty_target': '(ROOT)',
                          'span_text': 'original gold token/offset order, including repetitions',
                          'validation': 'parsed targets compared directly to original gold'}, 'sources': {}, 'splits': {}}
    for split in ('train', 'dev', 'test'):
        source = project_root / f'data/SciEvents/{split}.json'
        manifest['sources'][str(source.relative_to(project_root))] = hashlib.sha256(source.read_bytes()).hexdigest()
        output, stats = build_split(
            project_root, split, output_dir, semantic_path=semantic_path, appendix_schema=appendix_schema)
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
    parser.add_argument('--semantic-path', action='store_true',
                        help='build Region/SemanticDecision/EventType targets with the semantic prompt')
    parser.add_argument('--appendix-schema', action='store_true',
                        help='use Appendix F.2 exact schema text while keeping the C tree target')
    args = parser.parse_args()
    print(json.dumps(build_all(args.project_root, args.output_dir, semantic_path=args.semantic_path,
                               appendix_schema=args.appendix_schema), ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
