"""Exclusive online-dev run, verified training then held-out test or smoke reload."""
import argparse
from datetime import datetime, timezone
import importlib.util
import os
from pathlib import Path
import shutil
import sys
from types import SimpleNamespace

from online_dev_config import (CONFIG_DIR, HERE, PROJECT, STAGE, build_manifest,
                               jsonl, read, require, save, sha, source_files,
                               validate_manifest, verify_locks)


def test_diagnostics(m, prediction_path, merged_path):
    """Four separate test score modes; none feed training or checkpoint ranking."""
    from run_sorted_first import score_token
    from verify_sorted_first_run import fixed_evaluation_modules
    evaluation, _ = fixed_evaluation_modules()
    repair = STAGE/'scorer-repair-20261004/instance_scorer.py'
    spec = importlib.util.spec_from_file_location('online_instance_scorer', repair)
    instance = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(instance)
    gold = read(PROJECT/'data/SciEvents/test.json')
    g = [d['events'] for d in gold]
    merged = jsonl(merged_path)
    by_id = {d['document_id']: d['aligned_events'] for d in merged}
    require(len(by_id) == len(merged) and set(by_id) == {d['doc_id'] for d in gold},
            'Test merged document coverage differs')
    p = [by_id[d['doc_id']] for d in gold]
    token, raw_documents = score_token(Path(m['data_dir'])/'test.jsonl', prediction_path,
                                      PROJECT/'data/SciEvents/test.json')
    raw = [d['events'] for d in raw_documents]
    def compact(result):
        return {k: v for k, v in result.items() if k != 'matching'}
    return {'official_offset': evaluation.official_scores(g, p),
            'official_raw_token': token['official'],
            'instance_offset': compact(instance.score_documents(g, p, 'offset')),
            'instance_raw_token': compact(instance.score_documents(g, raw, 'token')),
            'raw_token_diagnostics': token['diagnostics']}


def smoke_reload(run_dir, m):
    """GPU child loads checkpoint independently after resident training exits."""
    from run_c_queue import run_stage
    cfg = read(run_dir/'training/config.json')
    output = run_dir/'smoke-reload/checkpoint-4.predictions.jsonl'
    run_stage([sys.executable, str(HERE/'predict_appendix_fixed.py'),
        '--data', cfg['dev_file'], '--output', str(output), '--model', m['model_path'],
        '--adapter', str(run_dir/'training/checkpoints/checkpoint-4'),
        '--batch-size', '4', '--seed', '42', '--max-new-tokens', '319'],
        run_dir/'smoke-reload.log')
    live = jsonl(run_dir/'training/feedback.jsonl')[-1]
    resident = jsonl(live['paths']['predictions'])
    loaded = jsonl(output)
    require([r['id'] for r in resident] == [r['id'] for r in loaded], 'Smoke reload IDs differ')
    require([r['raw_output'] for r in resident] == [r['raw_output'] for r in loaded],
            'Independent adapter reload differs from resident-model generation')
    completion = read(str(output)+'.complete.json')
    require(completion['count'] == len(resident) and completion['output_sha256'] == sha(output),
            'Smoke reload completion invalid')
    result = {'status': 'passed', 'count': len(resident), 'raw_outputs_exact': True,
              'adapter': str(run_dir/'training/checkpoints/checkpoint-4'), 'predictions': str(output)}
    save(run_dir/'smoke-reload-verification.json', result)
    return result


def run_after_training(run_dir, state, m):
    """Test is inaccessible until independent training and dev checks succeed."""
    from verify_sorted_first_online_dev import verify
    validation = verify(run_dir)
    save(run_dir/'training-verification.json', validation)
    cfg = read(run_dir/'training/config.json')
    terminal = read(run_dir/'training/train_metrics.json')
    state.update(best_step=terminal['best_step'], best_dev_score=terminal['best_score'],
                 last_step=terminal['global_step'], stop_reason=terminal['stop_reason'],
                 best_checkpoint=str(run_dir/'training/checkpoints'/f"checkpoint-{terminal['best_step']}"),
                 last_checkpoint=str(run_dir/'training/checkpoints'/f"checkpoint-{terminal['global_step']}"))
    if state['smoke']:
        state['current_activity'] = 'smoke_reload'
        save(run_dir/'run.json', state)
        state['smoke_reload'] = smoke_reload(run_dir, m)
    else:
        from appendix_fixed_eval import evaluate_checkpoint
        state['current_activity'] = 'test_dev_best'
        save(run_dir/'run.json', state)
        args = SimpleNamespace(data_dir=Path(m['data_dir']), document_data_dir=Path(m['document_data_dir']),
                               model=m['model_path'], batch_size=4, seed=42)
        test, pred, merged = evaluate_checkpoint(Path(state['best_checkpoint']), 'test', args, run_dir, 319)
        state.update(test_prediction=str(pred), test_merged=str(merged), test_official=test['official'],
                     test_complex=test['complex'], test_scores=test_diagnostics(m, pred, merged))
        require(state['test_official'] == state['test_scores']['official_offset'], 'Test scoring mismatch')
        save(run_dir/'run.json', state)
        save(run_dir/'final-verification.json', verify(run_dir, include_test=True))
    return state


def report(run_dir, state):
    training = read(run_dir/'training/train_metrics.json')
    rows = jsonl(run_dir/'training/feedback.jsonl')
    keys = ['TI_f1', 'TC_f1', 'AI_f1', 'AC_f1', 'EC_I_f1', 'EC_C_f1']
    lines = ['# Sorted1 online dev training', '',
        'Fresh original Qwen3-8B + new LoRA; unchanged AppendixNoCopy prompt/data/course. '
        'Online official offset dev AC controls plateau LR and stopping. '
        '3208 update maximum and new schedule differ from historical 4010-update cosine runs; '
        'this is a combined online-feedback + four-epoch-limit experiment, single seed42.', '',
        f"Status: {state['status']}; best step {state['best_step']}; last step {state['last_step']}; "
        f"stop reason {state['stop_reason']}. Smoke: {state['smoke']}.", '',
        '| dev step | TI | TC | AI | AC | EC_I | EC_C | LR after | bad count |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for row in rows:
        lines.append('| '+str(row['step'])+' | '+' | '.join(f'{row["official"][k]*100:.4f}' for k in keys)+
                     f" | {row['lr_after']:.8g} | {row['bad_count']} |")
    if not state['smoke']:
        lines += ['', '| test mode | TI | TC | AI | AC | EC_I | EC_C |',
                  '|---|---:|---:|---:|---:|---:|---:|']
        for mode in ['official_offset', 'official_raw_token', 'instance_offset', 'instance_raw_token']:
            scores = state['test_scores'][mode]
            if mode.startswith('instance'):
                scores = scores['metrics']
            lines.append('| '+mode+' | '+' | '.join(f'{scores[k]*100:.4f}' for k in keys)+' |')
        lines += ['', 'Instance scores are separate diagnostics; compare only identical scoring modes. '
                  'All 18 P/R/F1 and instance counts are retained in run.json.', '',
                  'Historical official-offset references (unchanged predictions): baseline AC41.88%, '
                  'Sorted1 AC41.98%; both dev-best2406. Do not attribute a difference only to early stopping.']
    lines += ['', f"Training + online-dev wall seconds: {training['wall_seconds']:.2f}; "
              f"peak allocated GiB: {training['peak_memory_bytes']/2**30:.3f}.", '',
              'Each dev step retains raw predictions, merged/aligned events, official metrics, '
              'diagnostics and complete safe state. Best and last adapters are separate.', '']
    (run_dir/'report.md').write_text('\n'.join(lines))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--manifest', type=Path, default=CONFIG_DIR/'manifest.json')
    p.add_argument('--smoke', action='store_true')
    args = p.parse_args()
    require('TMUX' in os.environ, 'GPU runner must run in tmux')
    sys.path.insert(0, str(STAGE/'scripts'))
    from run_c_queue import run_stage
    # Configuration created once from the frozen sources before either experiment.
    if args.manifest.exists():
        m = read(args.manifest)
    else:
        m = build_manifest()
        args.manifest.parent.mkdir(parents=True, exist_ok=True)
        with args.manifest.open('x') as stream:
            import json
            stream.write(json.dumps(m, indent=2, allow_nan=False)+'\n')
    validate_manifest(m)
    run_dir = args.output.resolve()
    run_dir.mkdir(parents=True, exist_ok=False)
    source_hashes = {str(p): sha(p) for p in source_files()}
    state = {'experiment': m['experiment'], 'status': 'running', 'current_activity': 'initialize',
             'smoke': args.smoke, 'started_at': datetime.now(timezone.utc).isoformat(),
             'manifest': str(args.manifest.resolve()), 'manifest_sha256': sha(args.manifest),
             'source_hashes': source_hashes, 'initialization': m['model_path'], 'init_adapter': None,
             'total_optimizer_steps': 4 if args.smoke else 3208,
             'eval_steps': 2 if args.smoke else 401, 'test_controls_training': False}
    for path in source_hashes:
        target = run_dir/'source_snapshot'/Path(path).relative_to(PROJECT)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
    save(run_dir/'run.json', state)
    try:
        state['current_activity'] = 'train_online_dev'
        save(run_dir/'run.json', state)
        run_stage([sys.executable, str(HERE/'train_sorted_first_online_dev.py'),
                   '--output', str(run_dir/'training'), '--manifest', str(args.manifest)] +
                  (['--smoke'] if args.smoke else []), run_dir/'training.log')
        verify_locks({**m['frozen_hashes'], **source_hashes})
        state['current_activity'] = 'verify_training_and_dev'
        save(run_dir/'run.json', state)
        state = run_after_training(run_dir, state, m)
        verify_locks({**m['frozen_hashes'], **source_hashes})
        state.update(status='completed', current_activity='completed',
                     finished_at=datetime.now(timezone.utc).isoformat())
        save(run_dir/'run.json', state)
        report(run_dir, state)
    except BaseException as exc:
        state.update(status='failed', error=str(exc), finished_at=datetime.now(timezone.utc).isoformat())
        save(run_dir/'run.json', state)
        raise


if __name__ == '__main__':
    main()
