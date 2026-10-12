"""Build audited Action supervision from corrected SciEvents sentence gold; no API."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import random
import re

from action_language import (canonical_events, event_multiset, load_schema, mention_text,
                             parse_actions, render_actions, render_tree)

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = ROOT/'tasks/stage2-event-tree/data/sentence-c-appendix-no-copy-gold-order-20261008'
DEFAULT_PROMPT = ROOT/'task/101202_action_prompt_v1.txt'
DEFAULT_IDS = ROOT/'tasks/stage2-event-tree/curriculum/sorted-first-gold-order-20261008/ordered-ids.txt'
DEFAULT_OUTPUT = ROOT/'tasks/action-experiment/data/action-v1-goldorder-20261012'


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()


def read_rows(path):
    with Path(path).open(encoding='utf-8') as handle:
        return [json.loads(line) for line in handle if line.strip()]


def validate_gold(row):
    tokens = row['tokens']
    lo, hi = row['target_bounds']
    if row['target_sentence'] != row['sentences'][row['sentence_index']] or row['target_sentence'] != ' '.join(tokens[lo:hi]):
        raise ValueError(f"Target sentence mismatch: {row['id']}")
    if row['n_events'] != len(row['target_events']):
        raise ValueError('Gold event count mismatch')
    for event in row['target_events']:
        for mention in [event['trigger'], *event['arguments']]:
            offsets = mention['offsets']
            if not offsets or any(type(i) is not int or i < 0 or i >= len(tokens) for i in offsets):
                raise ValueError(f"Invalid offsets: {row['id']}")
            if [tokens[i] for i in offsets] != mention['tokens'] or mention_text(mention) != mention['text']:
                raise ValueError(f"Gold mention text/offset mismatch: {row['id']}")
        if any(not lo <= i < hi for i in event['trigger']['offsets']):
            raise ValueError(f"Trigger outside target sentence: {row['id']}")


def make_row(row, template, schema):
    validate_gold(row)
    if template.count('{ABSTRACT}') != 1 or template.count('{TARGET_SENTENCE}') != 1:
        raise ValueError('Expected exactly two prompt placeholders')
    replacements = {'{ABSTRACT}': '\n'.join(f'[S{i+1}] {s}' for i,s in enumerate(row['sentences'])),
                    '{TARGET_SENTENCE}': f"[S{row['sentence_index']+1}] {row['target_sentence']}"}
    prompt = re.sub(r'\{ABSTRACT\}|\{TARGET_SENTENCE\}', lambda m: replacements[m[0]], template)
    events = canonical_events(row['target_events'], schema)
    actions = render_actions(events, schema)
    parsed = parse_actions(actions, schema)
    # Separate unordered comparison catches loss/merging beyond canonical ordering.
    original = [{'type': e['event_type'], 'trigger': mention_text(e['trigger']),
                 'arguments': [{'role': a['argument_type'], 'text': mention_text(a)} for a in e['arguments']]}
                for e in row['target_events']]
    if event_multiset(original) != event_multiset(parsed):
        raise ValueError(f"Gold/action multiset mismatch: {row['id']}")
    result = {k: v for k,v in row.items() if k not in ('prompt','completion','messages')}
    result.update(mode='sentence-action-v1',prompt=prompt,completion=actions,
                  action_events=events,compact_tree=render_tree(parsed))
    return result


def write_json(path, value):
    Path(path).write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')


def write_rows(path, rows):
    with Path(path).open('x',encoding='utf-8') as handle:
        for row in rows: handle.write(json.dumps(row,ensure_ascii=False,separators=(',',':'))+'\n')


def summary(rows):
    events = [e for row in rows for e in row['target_events']]
    mentions = [a for e in events for a in e['arguments']]
    return dict(rows=len(rows),events=len(events),arguments=len(mentions),
                none_rows=sum(not row['target_events'] for row in rows),
                multiple_event_rows=sum(row['n_events']>1 for row in rows),
                repeated_role_events=sum(len(e['arguments'])>len({a['argument_type'] for a in e['arguments']}) for e in events),
                shared_trigger_rows=sum(len({tuple(e['trigger']['offsets']) for e in row['target_events']})<row['n_events'] for row in rows),
                noncontinuous_or_nonmonotonic_arguments=sum(a['offsets']!=list(range(min(a['offsets']),max(a['offsets'])+1)) for a in mentions),
                types=dict(Counter(e['event_type'] for e in events)))


def sample_report(output, train, seed, count):
    chosen = random.Random(seed).sample(range(len(train)), count)
    samples = [{'train_index': index, **train[index]} for index in chosen]
    write_rows(output/'random15.jsonl',samples)
    lines = ['# 随机 Action 训练样本检查', '',
             f'从 train 均匀无放回抽取 {count} 条；seed={seed}。索引从0开始。', '',
             '每条包含完整学生prompt、原始gold（含offset）、动作和compact-tree。自动检查已通过；人工/独立审查另见review。', '']
    for number,row in enumerate(samples,1):
        lines.extend([f"## {number}. {row['id']}", '',f"train index={row['train_index']}；事件数={row['n_events']}", '',
                      '### 学生输入', '', '```text',row['prompt'],'```','',
                      '### 原始 Gold', '', '```json',json.dumps(row['target_events'],ensure_ascii=False,indent=2),'```','',
                      '### Action 答案', '', '```text',row['completion'],'```','',
                      '### 精简树', '', '```text',row['compact_tree'],'```',''])
    (output/'random15.md').write_text('\n'.join(lines),encoding='utf-8')
    return [dict(id=row['id'],train_index=row['train_index']) for row in samples]


def token_audit(all_rows, tokenizer_path):
    from transformers import AutoTokenizer
    def tokenizer_locks():
        return {str(p):sha(p) for p in Path(tokenizer_path).iterdir()
                if p.is_file() and p.name in ('tokenizer.json','tokenizer_config.json','vocab.json','merges.txt','added_tokens.json','special_tokens_map.json','chat_template.jinja')}
    tokenizer_inputs = tokenizer_locks()
    tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_path),local_files_only=True)
    lengths = {key: [] for key in ('prompt_tokens','completion_tokens','training_tokens')}
    for rows in all_rows.values():
        values = {key: [] for key in lengths}
        for row in rows:
            prompt = tokenizer.apply_chat_template([{'role':'user','content':row['prompt']}],tokenize=True,
                        add_generation_prompt=True,enable_thinking=False)
            full = tokenizer.apply_chat_template([{'role':'user','content':row['prompt']},
                        {'role':'assistant','content':row['completion']}],tokenize=True,
                        add_generation_prompt=False,enable_thinking=False)
            if full[:len(prompt)] != prompt or len(full) <= len(prompt):
                raise ValueError(f"Assistant token prefix mismatch: {row['id']}")
            values['prompt_tokens'].append(len(prompt))
            values['completion_tokens'].append(len(full)-len(prompt))
            values['training_tokens'].append(len(full))
        for key in lengths: lengths[key].extend(values[key])
    def stats(values):
        ordered=sorted(values)
        return {**{f'p{p}': ordered[min(len(ordered)-1,int((len(ordered)-1)*p/100))] for p in (50,95,99,100)},
                'count':len(values),'above_319':sum(v>319 for v in values),'above_4096':sum(v>4096 for v in values),'above_8192':sum(v>8192 for v in values)}
    if tokenizer_locks() != tokenizer_inputs:
        raise ValueError('Tokenizer files changed during audit')
    return dict(tokenizer=str(tokenizer_path),enable_thinking=False,completion_includes_chat_end=True,assistant_prefix_verified=True,
                tokenizer_inputs_sha256=tokenizer_inputs,lengths={key:stats(v) for key,v in lengths.items()})


def main():
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--source-dir',type=Path,default=DEFAULT_SOURCE)
    cli.add_argument('--prompt',type=Path,default=DEFAULT_PROMPT)
    cli.add_argument('--ordered-ids',type=Path,default=DEFAULT_IDS)
    cli.add_argument('--output',type=Path,default=DEFAULT_OUTPUT)
    cli.add_argument('--seed',type=int,default=42)
    cli.add_argument('--sample-count',type=int,default=15)
    cli.add_argument('--tokenizer',type=Path)
    args=cli.parse_args()
    if args.output.exists(): cli.error('Output exists; choose a new directory')
    inputs=[args.source_dir/f'{split}.jsonl' for split in ('train','dev','test')]
    inputs += [args.source_dir/'manifest.json',args.prompt,args.ordered_ids,ROOT/'data/SciEvents/ontology.json',Path(__file__),ROOT/'scripts/action_language.py']
    before={str(p.resolve()):sha(p) for p in inputs}
    schema=load_schema(args.prompt,ROOT/'data/SciEvents/ontology.json')
    template=args.prompt.read_text(encoding='utf-8')
    all_rows={}; info={}
    for split in ('train','dev','test'):
        source=read_rows(args.source_dir/f'{split}.jsonl')
        ids=[r['id'] for r in source]
        if not source or len(ids)!=len(set(ids)) or any(r['split']!=split for r in source):
            raise ValueError(f'Duplicate IDs/empty/wrong split: {split}')
        all_rows[split]=[make_row(row,template,schema) for row in source]
        info[split]=summary(all_rows[split])
    train=all_rows['train']; by_id={row['id']:row for row in train}
    ordered=args.ordered_ids.read_text().splitlines()
    if len(ordered)!=len(set(ordered)) or set(ordered)!=set(by_id):
        raise ValueError('Sorted1 IDs must exactly cover train')
    if not 0<args.sample_count<=len(train): raise ValueError('Invalid sample count')
    audit=token_audit(all_rows,args.tokenizer) if args.tokenizer else None
    if before!={str(p.resolve()):sha(p) for p in inputs}: raise ValueError('Inputs changed during build')
    args.output.mkdir(parents=True,exist_ok=False)
    for split,rows in all_rows.items(): write_rows(args.output/f'{split}.jsonl',rows)
    write_rows(args.output/'train-sorted.jsonl',[by_id[i] for i in ordered])
    (args.output/'ordered-ids.txt').write_text('\n'.join(ordered)+'\n')
    (args.output/'prompt_template.txt').write_text(template,encoding='utf-8')
    selected=sample_report(args.output,train,args.seed,args.sample_count)
    if audit: write_json(args.output/'token-audit.json',audit)
    output_hashes={str(p.resolve()):sha(p) for p in args.output.iterdir() if p.is_file()}
    manifest=dict(format='scievents-action-v1',created_at=datetime.now(timezone.utc).isoformat(),
                  input_sha256=before,output_sha256=output_hashes,splits=info,role_order=schema,
                  event_order='stable ascending min trigger offset; ties retain source order',
                  argument_order='prompt schema role order, then stable ascending min offset; internal order unchanged',
                  sorted1_exact_ids=True,all_rows_roundtrip_verified=True,gold_text_offsets_verified=True,
                  prompt_context_only=True,random_sample=dict(seed=args.seed,split='train',items=selected),
                  token_audit=audit is not None,
                  scoring_note='compact_tree is the new representation, not input to the legacy Trigger-child tree parser')
    write_json(args.output/'manifest.json',manifest)
    print(json.dumps(dict(output=str(args.output),splits=info,roundtrip='passed',token_audit=audit is not None),ensure_ascii=False))

if __name__=='__main__': main()
