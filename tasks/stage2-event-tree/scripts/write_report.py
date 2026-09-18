"""Summarize every Stage-2 run into report.md (official TI/TC/AI/AC/EC, alignment, complex subsets)."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import statistics


def load(path):
    return json.loads(Path(path).read_text(encoding='utf-8')) if Path(path).exists() else None


def pct(value):
    return '待运行' if value is None else f'{100 * value:.2f}'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    root = args.root
    rows, complex_rows, align_rows, groups = [], [], [], {}
    for run in sorted((root / 'runs').glob('*')):
        state = load(run / 'run.json')
        if not state:
            continue
        for metrics_path in sorted(run.glob('*.metrics.json')):
            name = metrics_path.name.replace('.metrics.json', '')
            split, _, variant = name.partition('.')
            m = load(metrics_path)
            o = m['official']
            label = f"{state['experiment']}" + (' (oracle hints)' if variant == 'oracle' else '')
            rows.append((label, state['seed'], split, state['status'], m['scope'], m['documents'],
                         o['TI_f1'], o['TC_f1'], o['AI_f1'], o['AC_f1'], o['EC_C_f1'], o['AC_p'], o['AC_r']))
            a = m['alignment']
            align_rows.append((label, state['seed'], split, a.get('invalid_documents', 0), a.get('hit_max_new_tokens', 0),
                               a.get('events', 0), a.get('unaligned_triggers', 0), a.get('arguments', 0), a.get('unaligned_arguments', 0),
                               a.get('parse_dropped_events', 0), a.get('parse_dropped_args', 0)))
            if split == 'test' and state['status'] == 'completed' and m['scope'] == 'full':
                groups.setdefault(label, []).append(o)
            if m.get('complex', {}).get('scores'):
                s = m['complex']['scores']
                complex_rows.append((label, state['seed'], s.get('Consecutive_AC_f1'), s.get('Overlap_TC_f1'), s.get('Overlap_AC_f1'),
                                     s.get('Reverse_AC_f1'), s.get('Subevent_TC_f1'), s.get('Subevent_AC_f1'), s.get('Subevent_EC_f1')))
    out = ['# Stage-2 Event Tree Generation 实验报告', '',
           f'自动汇总时间：{datetime.now(timezone.utc).isoformat()}。所有数值来自现存 JSON 结果，官方指标由 EXCEEDS 仓库 `utils.calculate_F1`（offset 口径）计算。', '',
           '模型：`/opt/models/Qwen3-8B` LoRA。T0 = 无 guidance；T1 = 全局 typed hints（flat）；T2 = typed hints 按类型分组 + Role→NuggetType schema。'
           '“oracle hints” 表示同一 adapter 在推理时改用 gold typed mentions 作为 hints，衡量 Stage-1 质量的上界。', '',
           '参考：EXCEEDS 论文/本机复现主模型 TI 75.3/75.9，TC 63.7/64.1，AI 45.0/44.9，AC 43.2/42.9，EC 48.3/49.1。', '',
           '## 主结果（F1，%）', '',
           '| 实验 | seed | split | 状态 | 范围 | 文档 | TI | TC | AI | AC | EC | AC P | AC R |',
           '| --- | ---: | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
    for r in rows:
        out.append('| ' + ' | '.join([r[0], str(r[1]), r[2], r[3], r[4], str(r[5])] + [pct(v) for v in r[6:]]) + ' |')
    out += ['', '## 解析与对齐诊断', '',
            '| 实验 | seed | split | invalid 文档 | 触顶截断 | 生成事件 | trigger 未对齐 | 生成论元 | 论元未对齐 | 解析丢弃事件 | 解析丢弃论元 |',
            '| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
    for r in align_rows:
        out.append('| ' + ' | '.join(str(x) for x in r) + ' |')
    if complex_rows:
        out += ['', '## 复杂场景（官方 eval_complex_quick_f1.py，test，F1 %）', '',
                '| 实验 | seed | Consecutive AC | Overlap TC | Overlap AC | Reverse AC | Subevent TC | Subevent AC | Subevent EC |',
                '| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
        for r in complex_rows:
            out.append('| ' + ' | '.join([r[0], str(r[1])] + [pct(v) for v in r[2:]]) + ' |')
    multi = {k: v for k, v in groups.items() if len(v) > 1}
    if multi:
        out += ['', '## test 多 seed 汇总（均值 ± 样本标准差）', '', '| 实验 | n | TI | TC | AI | AC | EC |', '| --- | ---: | ---: | ---: | ---: | ---: | ---: |']
        for label, items in multi.items():
            cells = []
            for key in ('TI_f1', 'TC_f1', 'AI_f1', 'AC_f1', 'EC_C_f1'):
                vals = [100 * i[key] for i in items]
                cells.append(f'{statistics.mean(vals):.2f} ± {statistics.stdev(vals):.2f}')
            out.append(f'| {label} | {len(items)} | ' + ' | '.join(cells) + ' |')
    (root / 'report.md').write_text('\n'.join(out) + '\n', encoding='utf-8')
    print(f'wrote {root / "report.md"} ({len(rows)} result rows)')


if __name__ == '__main__':
    main()
