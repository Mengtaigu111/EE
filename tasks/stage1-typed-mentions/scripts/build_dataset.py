#!/usr/bin/env python3
"""Build sentence-scoped typed gold without candidate propagation."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

from prompts import TYPES, make_prompt

PROJECT_ROOT = Path(__file__).resolve().parents[3]


def build_document(raw, normalized, split, mode):
    if mode not in ('S_G', 'SD_G'):
        raise ValueError(f'Unknown mode: {mode}')
    tokens = raw['document']
    if raw['doc_id'] != normalized['document_id'] or tokens != normalized['tokens']:
        raise ValueError('Raw and normalized document identity/tokens differ')
    owners, start = [], 0
    sentences = normalized['sentences']
    if len(sentences) != len(raw['sentences']):
        raise ValueError('Sentence count mismatch')
    for index, sentence in enumerate(sentences):
        words = raw['sentences'][index].split()
        end = start + len(words)
        if (sentence['start'] != start or sentence['end'] != end
                or sentence['sentence_index'] != index or tokens[start:end] != words
                or sentence['text'] != ' '.join(words)):
            raise ValueError(f'Sentence boundary/text mismatch: {raw["doc_id"]}:{index}')
        owners.extend([index] * len(words))
        start = end
    if start != len(tokens) or normalized['text'] != ' '.join(tokens):
        raise ValueError('Document text/token coverage mismatch')
    stats = Counter(documents=1, sentences=len(sentences))
    audit, seen, mentions = [], set(), [[] for _ in sentences]
    triggers = {tuple(sorted(set(e['trigger']['offsets']))) for e in raw['events']}
    annotation_order = 0
    for event in raw['events']:
        annotation_order += 1  # Raw traversal includes the event's trigger first.
        for arg in event['arguments']:
            order = annotation_order
            annotation_order += 1
            stats['arguments_all'] += 1
            kind = arg['nugget_type']
            if kind not in TYPES:
                if not kind.startswith('E-'):
                    raise ValueError(f'Unexpected nugget type: {kind}')
                stats['excluded_E_or_other'] += 1
                continue
            stats['typed_annotations'] += 1
            offsets = tuple(sorted(set(arg['offsets'])))
            if not offsets or any(type(x) is not int or x < 0 or x >= len(tokens) for x in offsets):
                raise ValueError(f'Invalid offsets: {raw["doc_id"]}: {arg}')
            key = (offsets, kind)
            if key in seen:
                stats['duplicate_typed_annotations'] += 1
                continue
            seen.add(key)
            stats['unique_typed_mentions'] += 1
            stats['unique_trigger_dual_typed'] += offsets in triggers
            text = ' '.join(tokens[x] for x in offsets)
            member_sentences = {owners[x] for x in offsets}
            entry = {'document_id': raw['doc_id'], 'annotation_order': order,
                     'type': kind, 'raw_text': arg['text'], 'raw_offsets': arg['offsets'],
                     'text': text, 'offsets': list(offsets)}
            if len(member_sentences) != 1:
                stats['cross_sentence'] += 1
                audit.append(dict(entry, reason='cross_sentence', sentence_indices=sorted(member_sentences)))
                continue
            if list(offsets) != arg['offsets'] or text != arg['text']:
                stats['local_canonicalized'] += 1
                audit.append(dict(entry, reason='canonicalized'))
            sentence_index = next(iter(member_sentences))
            mentions[sentence_index].append({'text': text, 'type': kind,
                                            'offsets': list(offsets), 'annotation_order': order})
            stats['retained_sentence_local'] += 1
            stats['type_' + kind] += 1
            stats['local_discontinuous'] += offsets != tuple(range(offsets[0], offsets[-1] + 1))
    rows = []
    for sentence, gold in zip(sentences, mentions):
        gold.sort(key=lambda m: (m['offsets'][0], m['offsets'][-1], m['annotation_order']))
        stats['empty_sentences'] += not gold
        completion = json.dumps({'mentions': [{'text': m['text'], 'type': m['type']} for m in gold]},
                                ensure_ascii=False, separators=(',', ':'))
        rows.append({'id': sentence['sentence_id'], 'document_id': raw['doc_id'],
                     'sentence_id': sentence['sentence_id'], 'sentence_index': sentence['sentence_index'],
                     'split': split, 'tokens': tokens[sentence['start']:sentence['end']],
                     'document_start': sentence['start'], 'target_sentence': sentence['text'],
                     'abstract': normalized['text'], 'prompt': make_prompt(sentence['text'],
                         normalized['text'] if mode == 'SD_G' else None),
                     'completion': completion, 'gold_mentions': gold})
    return rows, stats, audit


def select_examples(rows, count=4):
    """Greedy maximum new-type coverage, ties resolved by original train order."""
    if len(rows) < count:
        raise ValueError('Not enough training sentences for fixed examples')
    covered, selected, used = set(), [], set()
    for _ in range(count):
        index = max((i for i in range(len(rows)) if i not in used),
                    key=lambda i: (len({m['type'] for m in rows[i]['gold_mentions']} - covered), -i))
        selected.append(rows[index])
        used.add(index)
        covered.update(m['type'] for m in rows[index]['gold_mentions'])
    return selected


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8') as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, separators=(',', ':')) + '\n')


def build_all(project_root=PROJECT_ROOT, output_dir=None):
    project_root = Path(project_root)
    output_dir = Path(output_dir) if output_dir else project_root / 'tasks/stage1-typed-mentions/data'
    manifest = {'created_at': datetime.now(timezone.utc).isoformat(), 'sources': {}, 'splits': {},
                'rules': {'types': TYPES, 'deduplicate_by': ['offset_set', 'type'],
                          'cross_sentence': 'exclude entire mention, preserve in audit',
                          'text': 'join source tokens in sorted unique offset order',
                          'annotation_order': 'zero-based traversal: trigger then arguments, all events',
                          'few_shot': 'train only, greedy maximum new-type coverage; input-order tie break'}}
    train_rows = None
    for split in ('train', 'dev', 'test'):
        raw_path = project_root / f'data/SciEvents/{split}.json'
        normalized_path = project_root / f'tasks/llm-sft/normalized/{split}.documents.jsonl'
        gold_path = project_root / f'tasks/llm-sft/normalized/{split}.gold.jsonl'
        for path in (raw_path, normalized_path, gold_path):
            manifest['sources'][str(path.relative_to(project_root))] = hashlib.sha256(path.read_bytes()).hexdigest()
        raw_docs = json.loads(raw_path.read_text())
        normalized = [json.loads(line) for line in normalized_path.read_text().splitlines()]
        if len(raw_docs) != len(normalized):
            raise ValueError(f'Document count mismatch: {split}')
        counts, rows, audit = Counter(), [], []
        for raw, norm in zip(raw_docs, normalized):
            doc_rows, doc_counts, doc_audit = build_document(raw, norm, split, 'S_G')
            rows.extend(doc_rows)
            counts.update(doc_counts)
            audit.extend(doc_audit)
        if len({row['id'] for row in rows}) != len(rows):
            raise ValueError(f'Duplicate sentence IDs: {split}')
        write_jsonl(output_dir / 'S_G' / f'{split}.jsonl', rows)
        sd_rows = [dict(row, prompt=make_prompt(row['target_sentence'], row['abstract'])) for row in rows]
        write_jsonl(output_dir / 'SD_G' / f'{split}.jsonl', sd_rows)
        write_jsonl(output_dir / 'audit' / f'{split}.jsonl', audit)
        manifest['splits'][split] = dict(counts)
        if split == 'train':
            train_rows = sd_rows
    examples = select_examples(train_rows)
    (output_dir / 'few_shot.json').write_text(json.dumps(examples, ensure_ascii=False, indent=2) + '\n')
    manifest['few_shot_ids'] = [r['id'] for r in examples]
    manifest['few_shot_types'] = sorted({m['type'] for r in examples for m in r['gold_mentions']})
    manifest['outputs'] = {str(p.relative_to(output_dir)): hashlib.sha256(p.read_bytes()).hexdigest()
                           for p in sorted(output_dir.rglob('*.jsonl'))}
    manifest['outputs']['few_shot.json'] = hashlib.sha256((output_dir / 'few_shot.json').read_bytes()).hexdigest()
    (output_dir / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n')
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project-root', type=Path, default=PROJECT_ROOT)
    parser.add_argument('--output-dir', type=Path)
    args = parser.parse_args()
    print(json.dumps(build_all(args.project_root, args.output_dir), indent=2))


if __name__ == '__main__':
    main()
