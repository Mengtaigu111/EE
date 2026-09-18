#!/usr/bin/env python3
"""Regenerate a factual report from saved run metadata and evaluation artifacts."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
from statistics import mean, stdev

TYPES = ('OG', 'APP', 'MOD', 'FEA', 'TAK', 'DST', 'LIM', 'STR', 'WEA', 'DEG')
DEFAULT_ROOT = Path(__file__).resolve().parents[1]


def read_json(path):
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding='utf-8'))
    except json.JSONDecodeError:
        return None  # A runner may be writing the file; never invent missing values.


def number(value, percent=False):
    return '待运行' if value is None else f'{value * (100 if percent else 1):.2f}'


def link(path, root):
    return f'[{path.relative_to(root)}]({path.relative_to(root)})'


def formal_run(run, split, manifest):
    metric = run['metrics'].get(split)
    if run['meta'].get('status') not in ('completed', 'complete') or not metric:
        return False
    expected = (manifest or {}).get('splits', {}).get(split, {}).get('sentences')
    return (metric.get('scope') == 'full' and not metric.get('oracle', False)
            and (expected is None or metric.get('samples') == expected)
            and all(isinstance(metric.get(k, {}).get('f1'), (int, float)) for k in ('typed', 'untyped')))


def render_report(root):
    root = Path(root)
    manifest = read_json(root / 'data/manifest.json')
    runs = []
    for path in sorted((root / 'runs').glob('*/run.json')):
        metadata = read_json(path)
        if metadata:
            runs.append({'path': path.parent, 'meta': metadata,
                         'metrics': {s: read_json(path.parent / f'{s}.metrics.json') for s in ('dev', 'test')}})
    lines = ['# Stage-1 Typed Mention Extraction 实验报告', '',
             f'自动汇总时间：{datetime.now(timezone.utc).isoformat()}。所有数值来自现存 JSON 结果；缺失或未写完的文件标为待运行。', '',
             '模型：`/opt/models/Qwen3-8B`。E1 为 S+D LoRA 主实验；E1-S 为 S LoRA 对照；E0 为不训练的 S+D 四例格式检查。', '',
             '## 数据与口径', '']
    if manifest:
        lines.extend(['| split | 文档 | 句子 | 句内 typed mentions | 空句 | 跨句排除 | 句内不连续 | 重复标注去除 |',
                      '| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |'])
        for split, c in manifest['splits'].items():
            keys = ('documents', 'sentences', 'retained_sentence_local', 'empty_sentences',
                    'cross_sentence', 'local_discontinuous', 'duplicate_typed_annotations')
            lines.append('| ' + split + ' | ' + ' | '.join(str(c.get(k, 0)) for k in keys) + ' |')
        lines.extend(['', f"来源、输出 SHA256 与规则：{link(root / 'data/manifest.json', root)}。四例仅来自 train，覆盖类型：{', '.join(manifest.get('few_shot_types', []))}。", ''])
    else:
        lines.extend(['数据构建统计：待运行。', ''])
    lines.extend(['Gold 类型仅取原始 `events[].arguments[].nugget_type` 的 10 类；按 offset 集合和类型去重，同 span 多类型保留。'
                  '纯 trigger 和 E-* 排除，trigger 兼任普通类型保留。跨句整条排除，空句保留；不连续和乱序标注按原文 token 顺序输出，审计见 `data/audit/`。', '',
                  '评测为文档 token offset 集合 exact match；typed 额外要求类型相同。unaligned 计 FP，invalid 预测为空。'
                  'invalid rate 分母是句数，unaligned rate 分母是解析出的 mentions 数量。P/R/F1、recall 和 rate 均以百分数展示。', '',
                  '## E0 / E1 / E1-S 主结果', '',
                  'E0 仅评估 dev 前 32 句的 sanity 子集，不能与完整 dev/test 分数直接比较。smoke 仅验证执行链。'
                  '未完成的 run 即使已有部分指标，也不会进入正式结论或三 seed 汇总。', '',
                  '| 实验 | seed | split | 状态 / 范围 | 句数 | Untyped P | R | F1 | Typed P | R | F1 |',
                  '| --- | ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |'])
    observed = set()
    for run in runs:
        meta = run['meta']; experiment = meta.get('experiment', 'unknown'); seed = meta.get('seed', '?')
        observed.add((experiment, seed))
        for split in ('dev', 'test'):
            if experiment == 'E0' and split == 'test':
                continue
            metric = run['metrics'][split]
            values = [number((metric or {}).get(kind, {}).get(key), True)
                      for kind in ('untyped', 'typed') for key in ('precision', 'recall', 'f1')]
            status = str(meta.get('status', 'unknown')) + ' / ' + str((metric or {}).get('scope', '待运行'))
            lines.append(f"| {experiment} | {seed} | {split} | {status} | {(metric or {}).get('samples', '待运行')} | " + ' | '.join(values) + ' |')
    for experiment, seed in [('E0', 42), ('E1', 42), ('E1-S', 42), ('E1', 43), ('E1', 44)]:
        if (experiment, seed) not in observed:
            lines.append(f'| {experiment} | {seed} | dev' + ('/test' if experiment != 'E0' else ' 子集32句') + ' | 待运行 | 待运行 | ' + ' | '.join(['待运行'] * 6) + ' |')
    lines.extend(['', '## 类型召回与格式诊断', ''])
    for run in runs:
        for split, metric in run['metrics'].items():
            if not metric:
                continue
            label = f"{run['path'].name} / {split}"
            lines.extend([f'### {label}', '', link(run['path'] / f'{split}.metrics.json', root), '',
                          '| ' + ' | '.join(TYPES) + ' |', '| ' + ' | '.join(['---:'] * len(TYPES)) + ' |',
                          '| ' + ' | '.join('待运行' if t not in metric.get('per_type', {})
                                            else 'N/A (gold=0)' if metric['per_type'][t].get('recall') is None
                                            else number(metric['per_type'][t]['recall'], True) for t in TYPES) + ' |', '',
                          f"invalid rate：{number(metric.get('invalid_rate'), True)}%；unaligned rate：{number(metric.get('unaligned_rate'), True)}%；"
                          f"平均预测 mentions/句：{number(metric.get('average_predicted_mentions_per_sentence'))}。", '',
                          '错误分类：' + '；'.join(f"{side} " + ', '.join(f'{k}={v}' for k, v in counts.items())
                                                for side, counts in metric.get('errors', {}).items()) + '。', ''])
            error_path = run['path'] / f'{split}.metrics.errors.jsonl'
            if not error_path.exists():
                error_path = run['path'] / f'{split}.errors.jsonl'
            if error_path.exists():
                lines.extend([f'逐句错误：{link(error_path, root)}。以下按文件顺序展示最多两条有错误的句子，非随机抽样。', ''])
                shown = 0
                with error_path.open() as stream:
                    for line in stream:
                        try:
                            row = json.loads(line)
                        except json.JSONDecodeError:
                            continue
                        if not row.get('errors'):
                            continue
                        sentence = str(row.get('target_sentence', '')).replace('\n', ' ')
                        lines.extend([f"- `{row['id']}`：{sentence}", '',
                                      '  错误：' + json.dumps(row['errors'][:3], ensure_ascii=False) +
                                      ('（该句其余错误见原文件）' if len(row['errors']) > 3 else '') + '。', ''])
                        shown += 1
                        if shown == 2:
                            break
    if not any(m for r in runs for m in r['metrics'].values()):
        lines.extend(['类型召回、invalid / unaligned、平均 mentions 与 FN / FP：待运行。', ''])
    lines.extend(['## E1 test 三 seed 汇总', '',
                  '每个 seed 取路径排序最后的 completed/complete、完整 test run；保留上方所有运行记录。'
                  '均值和样本标准差使用 seeds 42、43、44（n=3，标准差分母 n−1），不是置信区间。', ''])
    selected = {}
    for run in runs:
        if run['meta'].get('experiment') == 'E1' and formal_run(run, 'test', manifest):
            selected[run['meta'].get('seed')] = run
    if all(seed in selected for seed in (42, 43, 44)):
        lines.extend(['| 指标 | 均值 | 样本标准差 |', '| --- | ---: | ---: |'])
        for kind in ('untyped', 'typed'):
            for key in ('precision', 'recall', 'f1'):
                values = [selected[seed]['metrics']['test'][kind][key] * 100 for seed in (42, 43, 44)]
                lines.append(f'| {kind} {key} | {mean(values):.2f} | {stdev(values):.2f} |')
        lines.extend(['', '使用运行：' + '、'.join(link(selected[s]['path'] / 'run.json', root) for s in (42, 43, 44)) + '。', ''])
    else:
        lines.extend(['尚不足三个已完成的完整 test seed；均值与样本标准差待运行。', ''])
    lines.extend(['## Gold-text roundtrip 诊断', '',
                  '将 gold text/type 送入同一确定性对齐器，检查文本接口的信息损失。'
                  '它不是模型性能的数学上界：模型可能输出不同顺序或文本而改变映射。重复文本、exact-first 干扰及不连续跨度会导致 roundtrip 错误。', '',
                  '| split | Untyped P | R | F1 | Typed P | R | F1 |',
                  '| --- | ---: | ---: | ---: | ---: | ---: | ---: |'])
    for split in ('train', 'dev', 'test'):
        path = root / 'verification' / f'oracle-{split}.json'; metric = read_json(path)
        values = [number((metric or {}).get(kind, {}).get(key), True)
                  for kind in ('untyped', 'typed') for key in ('precision', 'recall', 'f1')]
        label = link(path, root) if metric else split
        lines.append('| ' + label + ' | ' + ' | '.join(values) + ' |')
    lines.extend(['', '## Q1–Q3', ''])
    primary = selected.get(42)
    if primary:
        metric = primary['metrics']['test']
        lines.append(f"Q1：E1 seed42 完整 test typed F1={number(metric['typed']['f1'], True)}%，untyped F1={number(metric['untyped']['f1'], True)}%。"
                     '是否满足应用所需的“高质量”仍取决于召回要求，不能仅凭单个总分认定。')
    else:
        lines.append('Q1：主实验完整 test 结果待运行，暂不能判断抽取质量。')
    baseline = [r for r in runs if r['meta'].get('experiment') == 'E1-S' and r['meta'].get('seed') == 42
                and formal_run(r, 'test', manifest)]
    if primary and baseline:
        delta = 100 * (primary['metrics']['test']['typed']['f1'] - baseline[-1]['metrics']['test']['typed']['f1'])
        lines.extend(['', f'Q2：seed42 中 S+D 相对 S 的 typed F1 差为 {delta:+.2f} 个百分点。'
                      'E1-S 只有一个 seed，该差异仅是观察结果，不能判断统计显著性或稳定性。'])
    else:
        lines.extend(['', 'Q2：E1 与 E1-S 的完整 test 配对结果待运行，尚不能判断 abstract context 的帮助。'])
    if primary:
        fn = primary['metrics']['test'].get('errors', {}).get('FN', {})
        if fn and sum(fn.values()):
            dominant = max(fn, key=fn.get)
            lines.extend(['', f'Q3：E1 seed42 的 FN 中 {dominant} 最多（{fn[dominant]} 条）。'
                          '分类优先顺序是 TYPE_ONLY → BOUNDARY → MISSING；需结合逐句错误和 roundtrip 诊断解释来源。'])
        else:
            lines.extend(['', 'Q3：该主实验没有已记录的 FN，详细 FP 与样例见上文。'])
    else:
        lines.extend(['', 'Q3：主实验错误分析待运行。'])
    lines.extend(['', '## 运行证据', ''])
    if not runs:
        lines.append('运行目录待创建。')
    for run in runs:
        lines.append(f"- {link(run['path'] / 'run.json', root)}：{run['meta'].get('status', 'unknown')}。配置、训练日志、推理原文与错误 JSONL 位于同一运行目录。")
    return '\n'.join(lines) + '\n'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=DEFAULT_ROOT)
    args = parser.parse_args()
    args.root.mkdir(parents=True, exist_ok=True)
    output = args.root / 'report.md'
    temporary = output.with_suffix('.md.tmp')
    temporary.write_text(render_report(args.root), encoding='utf-8')
    temporary.replace(output)
    print(output)


if __name__ == '__main__':
    main()
