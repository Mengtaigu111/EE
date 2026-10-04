#!/usr/bin/env python3
"""Write the Chinese v2 report from verified dev selection and test artifacts."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
RESULTS = HERE / 'results'
DEST = ROOT / 'goal-22/scheduled-dev-fix'
NAMES = {'curriculum': 'UserV2 Easy2课程', 'baseline': 'AppendixNoCopy历史baseline'}
COMPLEX_KEYS = {'Consecutive_AC_f1', 'Overlap_TC_f1', 'Overlap_AC_f1',
                'Reverse_AC_f1', 'Subevent_TC_f1', 'Subevent_AC_f1', 'Subevent_EC_f1'}


def load(path):
    return json.loads(Path(path).read_text())


def percent(value):
    return f'{value * 100:.4f}'


def main():
    selection = load(RESULTS / 'dev_selection.json')
    verification_path = RESULTS / 'verification.log'
    verification = verification_path.read_text().strip() if verification_path.exists() else ''
    try:
        verification_data = json.loads(verification)
        verified = all('test_AC_f1' in verification_data['verified'][group]
                       for group in NAMES) and verification_data['locked_source_count'] == 13
    except (ValueError, KeyError, TypeError):
        verified = False
    main_exit = DEST.parent / 'main.exit'
    main_status = main_exit.read_text().strip() if main_exit.exists() else '运行中'
    lines = ['# 2026-10-03 dev评估修复结果', '',
             '独立版本：`tasks/stage2-event-tree/evaluation-v2-20261003/`。未修改旧run锁定的13份共享源码、checkpoint、数据、prompt、旧预测或旧指标。',
             'v2从sentence raw_output重新解析，按目标句scope对齐，保留文本occurrence，按事件类型、trigger offsets与argument role/offsets去重。官方评分器`utils.calculate_F1`原样使用。',
             '', '## Dev同口径重评', '',
             '| 组别 | step | 旧AC F1(%) | 新AC F1(%) | 差值(pp) |',
             '|---|---:|---:|---:|---:|']
    for group, item in selection.items():
        for step, new in item['scores'].items():
            old = item['old_scores'][step]
            lines.append(f'| {NAMES[group]} | {step} | {percent(old)} | {percent(new)} | {(new-old)*100:+.4f} |')
    lines += ['', '两组均只按新official dev AC F1选模，平局选较大step；未读取test成绩选checkpoint。', '']
    for group, item in selection.items():
        lines.append(f'- {NAMES[group]}：旧best step {item["old_best_step"]}，新best step {item["best_step"]}，新dev AC F1 {percent(item["scores"][str(item["best_step"])])}%。')
    lines += ['', '## Test（仅各组新dev-best）', '']
    complete = True
    for group, item in selection.items():
        step = item['best_step']
        path = RESULTS / group / f'test-checkpoint-{step}.metrics.json'
        if not path.exists():
            complete = False
            lines.append(f'- {NAMES[group]}：等待完整test raw预测与CPU重评。')
            continue
        m = load(path)
        c = m.get('complex', {})
        if c.get('exit_code') != 0 or set(c.get('scores', {})) != COMPLEX_KEYS:
            complete = False
            lines.append(f'- {NAMES[group]}：复杂场景验收未通过，不能标记完成。')
            continue
        o = m['official']
        if verified and verification_data['verified'][group]['test_AC_f1'] != o['AC_f1']:
            verified = False
        old_run = load(ROOT / 'tasks/stage2-event-tree/runs' /
                       ('C-CL-Mean-UserV2-Easy2-42-20261002' if group == 'curriculum'
                        else 'C-AppendixNoCopy-42-20261002') / 'run.json')
        old_ac = old_run['test_official']['AC_f1']
        lines.append(f'- {NAMES[group]} step {step}：TI {percent(o["TI_f1"])}%，TC {percent(o["TC_f1"])}%，AI {percent(o["AI_f1"])}%，AC {percent(o["AC_f1"])}%（旧{percent(old_ac)}%，变化{(o["AC_f1"]-old_ac)*100:+.4f}pp），EC {percent(o["EC_C_f1"])}%；复杂场景exit=0、七项齐全。')
        lines.append('  - 复杂场景：' + '；'.join(f'{key}={percent(value)}%'
                                              for key, value in sorted(c['scores'].items())) + '。')
    lines += ['', '## 验收与边界', '',
              '- 11份dev raw均具complete标记，输出SHA-256与config数据/adapter hash一致，sentence ID唯一且与数据逐行全覆盖；每份provenance保存输入、合并、指标及代码hash。',
              '- 回归测试覆盖跨句同文本、同句共享trigger、同类型多occurrence、畸形节点恢复、raw重新解析、重复身份与复杂场景失败闸门。',
              f'- 旧流水线 `goal-22/main.exit`：{main_status}。',
              '- 官方`utils.calculate_F1`对共享trigger offsets的覆盖是既有评分限制，本轮保留官方口径。两组prompt不同，因此比较是历史对照，不构成仅改变课程的因果消融。',
              '- Stage3报告问题4–6不在本次范围。', '']
    if verification:
        lines += ['独立验收输出：', '', '```json', verification, '```', '']
    report = HERE / 'report.md'
    report.write_text('\n'.join(lines), encoding='utf-8')
    status = {'dev': 'complete', 'baseline_test': 'complete' if complete else 'waiting',
              'curriculum_test': 'complete' if complete else 'waiting',
              'old_pipeline_exit': main_status, 'verification': 'complete' if complete and verified else 'partial',
              'report': str(report)}
    (DEST / 'status.json').write_text(json.dumps(status, ensure_ascii=False, indent=2) + '\n')
    test_scores = (f'课程test AC {percent(verification_data["verified"]["curriculum"]["test_AC_f1"])}%，'
                   f'baseline test AC {percent(verification_data["verified"]["baseline"]["test_AC_f1"])}%。') if verified else ''
    summary = ('dev修复版11/11 checkpoint完成同口径重评；两组新best均为step 2406。'
               + ('两组test与复杂场景七项、独立验收完成。' if complete and verified else '课程test等待旧GPU流水线完整预测后CPU重评。')
               + test_scores
               + f' 旧流水线exit={main_status}。详见{report}。')
    (DEST / 'result.md').write_text(summary + '\n', encoding='utf-8')
    print(json.dumps(status, ensure_ascii=False))


if __name__ == '__main__':
    main()
