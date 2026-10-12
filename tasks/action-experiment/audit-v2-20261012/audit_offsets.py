"""Read-only full-corpus offset preservation audit against raw SciEvents gold."""
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
DATA = ROOT / 'tasks/action-experiment/data/action-v2-goldorder-20261012'
sys.path[:0] = [str(ROOT / 'scripts'), str(ROOT / 'tasks/action-experiment')]
from action_language import load_schema, parse_actions, event_multiset
from action_evaluation import ordered_match


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def occurrences(needle, tokens):
    n = len(needle)
    return [list(range(i, i+n)) for i in range(len(tokens)-n+1) if tokens[i:i+n] == needle]


def text_event(e):
    return dict(type=e['event_type'], trigger=' '.join(e['trigger']['tokens']),
                arguments=[dict(role=a['argument_type'], text=' '.join(a['tokens'])) for a in e['arguments']])


def run():
    manifest = json.loads((DATA / 'manifest.json').read_text())
    source_paths = [ROOT / f'data/SciEvents/{split}.json' for split in ('train', 'dev', 'test')]
    locks = {**manifest['input_sha256'], **manifest['output_sha256']}
    locks.update({str(p): sha(p) for p in source_paths})
    for path, expected in locks.items():
        assert sha(path) == expected, path
    schema = load_schema(DATA / 'prompt_template.txt', ROOT / 'data/SciEvents/ontology.json')
    totals = Counter(); splits = {}; examples = {}; all_failures = []
    for split in ('train', 'dev', 'test'):
        docs = {d['doc_id']: d for d in json.loads((ROOT / f'data/SciEvents/{split}.json').read_text())}
        stats = Counter()
        for line in (DATA / f'{split}.jsonl').open():
            row = json.loads(line); doc = docs[row['document_id']]; lo, hi = row['target_bounds']
            gold = [e for e in doc['events'] if e['trigger']['offsets'] and all(lo <= i < hi for i in e['trigger']['offsets'])]
            try:
                assert row['tokens'] == doc['document']
                assert row['target_events'] == gold
                parsed = parse_actions(row['completion'], schema)
                assert event_multiset(parsed) == event_multiset([text_event(e) for e in gold])
                stats['rows'] += 1; stats['events'] += len(gold)
                # All multiplicities must remain, even when text collapses distinct offsets.
                grouped = defaultdict(set)
                for e in gold:
                    sig = (e['event_type'], ' '.join(e['trigger']['tokens']))
                    grouped[sig].add(tuple(e['trigger']['offsets']))
                    role_groups = defaultdict(set)
                    for a in e['arguments']:
                        role_groups[(a['argument_type'], ' '.join(a['tokens']))].add(tuple(a['offsets']))
                    stats['same_role_text_distinct_offset_groups'] += sum(len(v) > 1 for v in role_groups.values())
                    for kind, m in [('trigger', e['trigger'])] + [('argument', a) for a in e['arguments']]:
                        offsets = m['offsets']; tokens = m['tokens']; text = ' '.join(tokens)
                        assert offsets and all(type(i) is int and 0 <= i < len(doc['document']) for i in offsets)
                        assert [doc['document'][i] for i in offsets] == tokens
                        assert m['text'] == text
                        stats[kind+'s'] += 1
                        nonmonotonic = any(a >= b for a, b in zip(offsets, offsets[1:]))
                        noncontiguous = offsets != list(range(min(offsets), max(offsets)+1))
                        candidates = occurrences(tokens, row['tokens'])
                        diagnostic = ordered_match(text, row['target_sentence'].split() if kind == 'trigger' else row['tokens'])
                        stats[kind+'_noncontiguous'] += noncontiguous
                        stats[kind+'_nonmonotonic'] += nonmonotonic
                        stats[kind+'_multiple_contiguous_text_occurrences'] += len(candidates) > 1
                        stats[kind+'_gold_ordered_diagnostic_false'] += not diagnostic
                        category = 'nonmonotonic' if nonmonotonic else ('discontinuous' if noncontiguous else ('repeated_text' if len(candidates) > 1 else None))
                        if category and category not in examples:
                            examples[category] = dict(split=split, id=row['id'], document_id=row['document_id'],
                                mention_kind=kind, event_type=e['event_type'], role=m.get('argument_type'),
                                offsets=offsets, text=text, tokens_at_offsets=[row['tokens'][i] for i in offsets],
                                contiguous_candidate_offsets=candidates[:6], ordered_diagnostic=diagnostic,
                                target_sentence=row['target_sentence'], action=row['completion'])
                groups = sum(len(v) > 1 for v in grouped.values())
                stats['same_type_trigger_text_distinct_offset_groups'] += groups
                if groups and 'duplicate_event_text' not in examples:
                    examples['duplicate_event_text'] = dict(split=split, id=row['id'], gold_events=gold, action=row['completion'])
            except (AssertionError, ValueError, KeyError) as error:
                all_failures.append(dict(split=split, id=row['id'], error=repr(error)))
        splits[split] = dict(stats); totals.update(stats)
    for path, expected in locks.items():
        assert sha(path) == expected, path
    result = dict(status='passed' if not all_failures else 'failed', totals=dict(totals), splits=splits,
                  failures=all_failures, all_raw_offsets_and_mention_word_order_preserved=not all_failures,
                  action_event_and_role_multiplicity_preserved=not all_failures,
                  immutable_inputs_verified_before_and_after=True, source_sha256=locks,
                  examples=examples,
                  limitation='Text-only Action/Tree does not encode offsets and cannot uniquely reconstruct repeated occurrences or arbitrary nonmonotonic fragments. ordered_match is a subsequence-presence diagnostic, not an offset aligner; it never removes or changes predictions/rewards.')
    (OUT / 'offset-audit.json').write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n')
    lines = ['# 全量原始 offset 与 Action 保留性审计', '',
             f"结果：{result['status']}。核对 {totals['rows']} 条句子、{totals['events']} 个事件、{totals['arguments']} 个论元；失败 {len(all_failures)} 条。",
             '', '所有 mention 按原始 offset **给定顺序**索引 raw document 后，均与 gold tokens/text 相等；不排序片段、不合并相同文本事件或论元。Action 的事件/论元 multiset 与原始 gold 完全相等。', '',
             '## 计数', '', '```json', json.dumps(dict(totals), ensure_ascii=False, indent=2), '```', '',
             '## 定位边界', '',
             '当前评分基于文本存在匹配，不需要把预测文本逆推到唯一 offset。训练数据仍保留原始 gold offsets；Action/Tree 只输出文本，因此重复文本、多种等效片段选择无法唯一恢复坐标。该信息丢失属于格式表达能力的限制，不能通过选择第一个匹配位置修复。', '',
             '`ordered_match` 只判断顺序子序列是否存在，返回布尔值，既不返回 offset，也不保证连续性或唯一性。非单调 gold 可能被诊断为无序列匹配，这不等于 gold/action 错误；该诊断不会删预测，也不影响 presence 分数或奖励。', '',
             '若后续要求精确 span/offset 指标，应单独增加片段坐标输出协议与对应监督；不能声称现在纯文本 Action 能唯一复原全部原始 offset。', '', '## 真实样本']
    for category, example in examples.items():
        lines.extend(['', '### '+category, '', '```json', json.dumps({k:v for k,v in example.items() if k != 'action'}, ensure_ascii=False, indent=2), '```', '', '```text', example['action'], '```'])
    (OUT / 'offset-audit.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(dict(status=result['status'], totals=dict(totals), failures=len(all_failures)), ensure_ascii=False))
    if all_failures:
        raise SystemExit(1)


if __name__ == '__main__':
    run()
