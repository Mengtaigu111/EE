"""Main mean-percentile curriculum: continuous budget, exact approved UserV2 prompt."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
import shutil
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
PROJECT = ROOT.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / 'scripts'))
from run_c_queue import evaluate_checkpoint, run_stage, save

BASELINE = ROOT / 'runs/C-AppendixNoCopy-42-20261002'


def checkpoint_steps(total, boundary, cadence):
    return sorted(set(range(cadence, total + 1, cadence)) | {boundary, total})


def select_checkpoint(results):
    return max(results, key=lambda r: (r['AC_f1'], r['step']))


def verify_budget(total, stage1, final):
    if not 0 < stage1 < total or final != total:
        raise ValueError('Curriculum phases must be positive and sum to baseline budget')
    return stage1, final - stage1


def validate_schedule(manifest, cfg, total):
    batch = cfg['microbatch'] * cfg['gradient_accumulation']
    expected = 2 * math.ceil(manifest['stage1_total_count'] / batch)
    if (manifest['stage1_epoch_target'] != 2 or manifest['stage1_optimizer_steps'] != expected
            or manifest['stage2_optimizer_steps'] != total - expected):
        raise ValueError('Expected exact two-epoch Stage1 and remaining full-data budget')
    verify_budget(total, expected, total)
    return expected


def smoke_schedule():
    return 6, 2, 2


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def report(run_dir, state, manifest, publish=True):
    baseline = read(state['baseline_run'])
    baseline_dev = next(x for x in baseline['dev_checkpoints'] if x['checkpoint'] == baseline['best_checkpoint'])
    refs = [('AppendixNoCopy baseline (dev best)', Path(baseline_dev['metrics'])),
            ('Curriculum Stage 1 end', Path(state['stage1_dev_metrics'])),
            ('Curriculum final', Path(state['final_dev_metrics'])),
            ('Curriculum best checkpoint', Path(state['best_dev_metrics']))]
    lines = ['# UserV2 Easy2 mean curriculum training report', '',
        '用户指定 UserV2 prompt；复用原样本和 gold，仅替换 train/dev/test prompt。设置及预算参考 C-AppendixNoCopy-42-20261002。',
        '从原始 Qwen3-8B 初始化；无 Region / SemanticDecision 节点。单 seed，不提前假设 curriculum 有收益。', '',
        '## Q1. Data construction', '',]
    for key in ('B1_count', 'B2_count', 'root_total', 'root_stage1_count', 'stage1_total_count', 'stage2_total_count'):
        lines.append(f'- {key}: {manifest[key]}')
    lines += ['', 'Stage2 JSONL 与新 UserV2 full train 字节一致；全部 gold、样本和课程 IDs 与原数据一致，ROOT subset 原样复用。',
              'ROOT比例：' + ', '.join(f'{k}={manifest[k]*100:.6f}%' for k in
                ('full_train_root_ratio', 'stage1_root_ratio', 'stage2_root_ratio')), '',
              '## Q2. Training budget', '',]
    total = manifest['baseline_total_optimizer_steps']
    m1, m2 = read(run_dir / 'stage1/train_metrics.json'), read(run_dir / 'stage2/train_metrics.json')
    s1, s2 = verify_budget(total, m1['global_step'], m2['global_step'])
    lines += [f'Baseline={total}; Stage1={s1}; Stage2={s2}; total={s1+s2}，严格相等。',
              f'Cosine horizon={total}，warmup_ratio=0.03，仅从 global step0 开始一次。',
              '', '## Q3. Resume correctness', '',]
    evidence = read(run_dir / 'stage2/resume_evidence.json')
    lines += ['恢复证据：`stage2/resume_evidence.json`；optimizer/scheduler/RNG 采用 safetensors + JSON。',
              'bf16 无 GradScaler（N/A）；不升级共享环境，不跳过新 full-train 数据迭代器的旧阶段 batch。',
              '```json', json.dumps(evidence, ensure_ascii=False, indent=2), '```', '',
              '## Q4. Dev results', '', '| Run / Stage | TI | TC | AI | AC | EC |', '|---|---:|---:|---:|---:|---:|']
    keys = ['TI_f1', 'TC_f1', 'AI_f1', 'AC_f1', 'EC_C_f1']
    rows = []
    for label, path in refs:
        official = read(path)['official']
        rows.append(official)
        lines.append('| ' + label + ' | ' + ' | '.join(f'{official[k]*100:.4f}' for k in keys) + ' |')
    lines += ['', '全部为 dev 百分比；EC 使用官方 EC_C_f1。用户提供的74.84等是 test，不混入本表。',
              f"Best: {state['best_checkpoint']}, step={state['best_step']}, selection=official dev AC_f1（平局较大step）。",
              f"对照5个checkpoint；curriculum {len(state['dev_checkpoints'])}个（额外包含强制Stage1边界）。", '',
              '## Q5. Delta vs AppendixNoCopy', '',
              ', '.join(f'Δ{k.split("_")[0]}={(rows[-1][k]-rows[0][k])*100:+.4f} pp' for k in keys), '',
              'prompt 与 curriculum 同时变化；与历史 AppendixNoCopy 的差值不能全部归因于 curriculum。未自动重训新prompt baseline。', '',
              '## Q6. Training behavior', '',
              f"Stage1 loss={m1['phase_train_loss']:.6f}; Stage2 loss={m2['phase_train_loss']:.6f}（按本阶段updates校正Trainer汇总）。",
              f"边界最近loss: Stage1 last={m1['last_logged_loss']}; Stage2 first={m2['first_logged_loss']}。这是不同batch，单点跳变不作稳定性判断。",
              f"阶段collator读取: S1={m1['collated_examples']} examples/{m1['effective_collated_passes']:.4f} passes; S2={m2['collated_examples']} examples/{m2['effective_collated_passes']:.4f} passes（可能含一个prefetch batch）。",
              f"最佳checkpoint位于 {'Stage1' if state['best_step'] <= s1 else 'Stage2'}。",
              '主训练两阶段exit0；每个microbatch原始loss均检查finite，NaN/Inf会在optimizer更新前停止。未发生OOM；各阶段日志保留全部warnings。', '',
              '## Provenance', '',]
    for label, path in refs:
        lines.append(f'- {label}: `{path}`; SHA256 `{sha(path)}`')
    lines += [f'- Manifest: `{state.get("manifest", HERE / "mean_curriculum_manifest.json")}`',
              f'- Prompt: `{manifest["prompt_template_file"]}`; SHA256 `{manifest["prompt_sha256"]}`',
              f'- Run: `{run_dir / "run.json"}`', f'- Baseline run SHA256: `{sha(state["baseline_run"])}`', '',
              '## Test (dev-best only)', '', '| TI | TC | AI | AC | EC |', '|---:|---:|---:|---:|---:|']
    test = state['test_official']
    lines.append('| ' + ' | '.join(f'{test[k]*100:.4f}' for k in keys) + ' |')
    lines += ['', 'Test仅评估dev选出的一个checkpoint。', '']
    out = '\n'.join(lines)
    (run_dir / 'mean_curriculum_training_report.md').write_text(out)
    # Canonical report may be published once; refuse overwriting any earlier run.
    if publish:
        canonical = Path(state['manifest']).parent / 'mean_curriculum_training_report.md'
        with canonical.open('x') as stream:
            stream.write(out)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--baseline', type=Path, default=BASELINE)
    p.add_argument('--run-dir', type=Path, required=True)
    p.add_argument('--manifest', type=Path, default=HERE / 'user-v2-easy2-20261002/mean_curriculum_manifest.json')
    p.add_argument('--smoke', action='store_true')
    p.add_argument('--batch-size', type=int, default=4)
    a = p.parse_args()
    if 'TMUX' not in os.environ:
        raise RuntimeError('GPU execution must run in tmux')
    a.baseline = a.baseline.resolve()
    baseline = read(a.baseline / 'run.json')
    cfg = read(a.baseline / 'training/config.json')
    manifest = read(a.manifest)
    for path, expected in {**manifest['source_hashes'], **manifest['output_hashes']}.items():
        if sha(path) != expected:
            raise ValueError(f'Input provenance mismatch: {path}')
    if Path(manifest['baseline_run_file']).resolve() != (a.baseline / 'run.json').resolve():
        raise ValueError('Manifest must use same baseline run')
    total = manifest['baseline_total_optimizer_steps']
    if total != read(a.baseline / 'training/trainer_state.json')['global_step']:
        raise ValueError('Manifest/baseline actual budget mismatch')
    # Cadence is derived from actual baseline checkpoints, not assumed floor(N/batch).
    cadence = min(r['step'] for r in baseline['dev_checkpoints'])
    boundary = validate_schedule(manifest, cfg, total)
    if a.smoke:
        total, boundary, cadence = smoke_schedule()
    a.run_dir = a.run_dir.resolve()
    a.run_dir.mkdir(parents=True, exist_ok=False)
    state = {'experiment': 'C-CL-Mean-UserV2-Easy2', 'seed': cfg['seed'], 'status': 'running',
        'started_at': datetime.now(timezone.utc).isoformat(), 'smoke': a.smoke,
        'baseline_run': str(a.baseline / 'run.json'), 'baseline_sha256': sha(a.baseline / 'run.json'),
        'manifest': str(a.manifest.resolve()), 'manifest_sha256': sha(a.manifest),
        'total_optimizer_steps': total, 'stage1_optimizer_steps': boundary,
        'stage2_optimizer_steps': total-boundary, 'scheduler_horizon': total,
        'checkpoint_steps': checkpoint_steps(total, boundary, cadence),
        'prompt_source': manifest['source_train_file'],
        'prompt_template_file': manifest['prompt_template_file'], 'prompt_sha256': manifest['prompt_sha256'],
        'comparison_limitation': 'prompt and curriculum both changed relative to historical AppendixNoCopy',
        'initialization': cfg['model_path'], 'init_adapter': None}
    source_files = [HERE / x for x in ('run_user_v2_easy2_curriculum.py', 'train_user_v2_phase.py', 'curriculum_trainer.py')]
    source_files += [ROOT / 'scripts' / x for x in ('run_c_queue.py', 'model_utils.py', 'predict.py',
        'linearize.py', 'ontology.py', 'align_events.py', 'merge_sentence_predictions.py', 'evaluate.py')]
    source_files += [PROJECT / 'utils.py', PROJECT / 'eval_complex_quick_f1.py']
    state['source_code_hashes'] = {str(x): sha(x) for x in source_files}
    for file in source_files:
        target = a.run_dir / 'source_snapshot' / file.parent.name / file.name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(file, target)
    save(a.run_dir / 'run.json', state)
    eval_args = argparse.Namespace(data_dir=Path(manifest['data_dir']),
        document_data_dir=Path(baseline['document_data_dir']), model=cfg['model_path'],
        seed=cfg['seed'], batch_size=a.batch_size)
    try:
        for phase, stop in ((1, boundary), (2, total)):
            for path, expected in state['source_code_hashes'].items():
                if sha(path) != expected:
                    raise ValueError(f'Source changed during run: {path}')
            state['current_activity'] = f'train_stage{phase}'
            save(a.run_dir / 'run.json', state)
            output = a.run_dir / f'stage{phase}'
            data = Path(manifest[f'stage{phase}_data'])
            cmd = [sys.executable, str(HERE / 'train_user_v2_phase.py'), '--data', str(data),
                '--output', str(output), '--baseline', str(a.baseline), '--total-steps', str(total),
                '--stop-step', str(stop), '--save-steps', str(cadence)]
            if phase == 2:
                cmd += ['--resume', str(a.run_dir / 'stage1/checkpoints' / f'checkpoint-{boundary}')]
            if a.smoke:
                cmd += ['--smoke', '--limit', '32']
            run_stage(cmd, a.run_dir / f'stage{phase}_train.log')
            if not a.smoke and phase == 1:
                state['current_activity'] = 'evaluate_stage1_boundary'
                save(a.run_dir / 'run.json', state)
                cp = output / 'checkpoints' / f'checkpoint-{stop}'
                metrics, _, _ = evaluate_checkpoint(cp, 'dev', eval_args, a.run_dir, baseline['max_new_tokens'])
                state['stage1_dev_metrics'] = str(a.run_dir / 'checkpoint-eval' / f'dev-{cp.name}.metrics.json')
                state['stage1_dev_official'] = metrics['official']
                save(a.run_dir / 'run.json', state)
        m1, m2 = read(a.run_dir / 'stage1/train_metrics.json'), read(a.run_dir / 'stage2/train_metrics.json')
        verify_budget(total, m1['global_step'], m2['global_step'])
        if m1['phase_optimizer_steps'] != boundary or m2['phase_optimizer_steps'] != total-boundary:
            raise ValueError('Wrong phase updates')
        if not a.smoke and not math.isclose(read(a.run_dir / 'stage1/trainer_state.json')['epoch'], 2.0, abs_tol=1e-12):
            raise ValueError('Stage1 must finish exactly two epochs')
        state['resume_evidence'] = read(a.run_dir / 'stage2/resume_evidence.json')
        if not a.smoke:
            for path, expected in state['source_code_hashes'].items():
                if sha(path) != expected:
                    raise ValueError(f'Source changed during run: {path}')
            state['current_activity'] = 'evaluate_dev_checkpoints'
            save(a.run_dir / 'run.json', state)
            checkpoints = sorted([*(a.run_dir / 'stage1/checkpoints').glob('checkpoint-*'),
                                  *(a.run_dir / 'stage2/checkpoints').glob('checkpoint-*')],
                                 key=lambda x: int(x.name.split('-')[-1]))
            if [int(c.name.split('-')[-1]) for c in checkpoints] != state['checkpoint_steps']:
                raise ValueError('Unexpected saved checkpoints')
            dev = []
            for cp in checkpoints:
                step = int(cp.name.split('-')[-1])
                path = a.run_dir / 'checkpoint-eval' / f'dev-{cp.name}.metrics.json'
                if step == boundary:
                    metrics = read(path)
                else:
                    metrics, _, _ = evaluate_checkpoint(cp, 'dev', eval_args, a.run_dir, baseline['max_new_tokens'])
                dev.append({'checkpoint': str(cp), 'step': step, 'AC_f1': metrics['official']['AC_f1'], 'metrics': str(path)})
            best = select_checkpoint(dev)
            state.update(dev_checkpoints=dev, best_checkpoint=best['checkpoint'], best_step=best['step'],
                         best_dev_metrics=best['metrics'], final_dev_metrics=dev[-1]['metrics'])
            save(a.run_dir / 'run.json', state)
            test, prediction, merged = evaluate_checkpoint(Path(best['checkpoint']), 'test', eval_args, a.run_dir,
                                                           baseline['max_new_tokens'])
            state.update(test_official=test['official'], test_prediction=str(prediction), test_merged=str(merged))
            report(a.run_dir, state, manifest)
        state.update(status='completed', finished_at=datetime.now(timezone.utc).isoformat())
        save(a.run_dir / 'run.json', state)
    except Exception as exc:
        state.update(status='failed', error=repr(exc), failed_at=datetime.now(timezone.utc).isoformat())
        save(a.run_dir / 'run.json', state)
        raise
    print(json.dumps(state, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
