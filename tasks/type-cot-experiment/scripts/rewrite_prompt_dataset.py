"""Audit existing CoT targets and replace only prompts in a new dataset directory."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re

from parser import EVENT_CODES, parse_type_cot_output


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rewrite(source, template, output):
    if output.exists():
        raise ValueError(f'refusing to overwrite {output}')
    text = template.read_text()
    marker = '\nFull Abstract:\n'
    if text.count(marker) != 1:
        raise ValueError('template must contain exactly one Full Abstract section')
    instructions = text.split(marker)[0]
    splits, counts = {}, Counter()
    for split in ('train', 'dev', 'test'):
        rows = [json.loads(line) for line in (source / f'{split}.jsonl').read_text().splitlines() if line.strip()]
        for row in rows:
            gold = Counter(event['event_type'] for event in row['target_events'])
            parsed = parse_type_cot_output(row['completion'])
            if (parsed['source'] != 'inventory' or parsed['unknown_lines']
                    or parsed['counts'] != dict(gold)):
                raise ValueError(f'{row["id"]}: final inventory differs from gold')
            if split == 'train':
                completion = row['completion']
                reasoning = completion.split('Final Answer:')[0]
                explained = Counter(re.findall(r'Type:\s*<?(' + '|'.join(EVENT_CODES) + r')\b', reasoning))
                if explained != gold:
                    raise ValueError(f'{row["id"]}: reasoning counts differ from gold')
                if 'Reasoning:' not in reasoning or 'Event Family Reasoning:' not in reasoning:
                    raise ValueError(f'{row["id"]}: missing reasoning headings')
                if gold and 'Event Reasoning:' not in reasoning:
                    raise ValueError(f'{row["id"]}: missing event reasoning')
                counts['train_rows'] += 1
                counts['positive_rows' if gold else 'no_event_rows'] += 1
                counts['explained_events'] += sum(gold.values())
            sentences = row.get('sentences')
            if not sentences:
                raise ValueError(f'{row["id"]}: missing full-abstract sentences')
            index = row['sentence_index']
            if sentences[index] != row['target_sentence']:
                raise ValueError(f'{row["id"]}: target sentence differs from abstract')
            abstract = '\n'.join(f'[S{i + 1}] {sentence}' for i, sentence in enumerate(sentences))
            row['prompt'] = (instructions + marker + abstract +
                             f'\n\nTARGET SENTENCE [S{index + 1}]:\n' +
                             row['target_sentence'] + '\n\nType-COT output:')
        splits[split] = rows
    output.mkdir(parents=True)
    manifest = dict(created_at=datetime.now(timezone.utc).isoformat(),
                    source_dir=str(source.resolve()), prompt_template=str(template.resolve()),
                    prompt_sha256=sha256(template), audit=dict(counts), splits={},
                    policy='Only prompt is replaced; all other fields, including completion and gold, are unchanged.',
                    limits='Full structural/count audit; semantic compatibility reviewed on samples, not exhaustively proven.')
    for split, rows in splits.items():
        path = output / f'{split}.jsonl'
        path.write_text(''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in rows))
        manifest['splits'][split] = dict(rows=len(rows), source_sha256=sha256(source / f'{split}.jsonl'),
                                        output_sha256=sha256(path))
    (output / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n')
    return manifest


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--template', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(rewrite(args.source, args.template, args.output), ensure_ascii=False, indent=2))
