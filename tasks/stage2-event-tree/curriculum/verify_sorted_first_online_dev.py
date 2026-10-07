"""Independent replay of online feedback, actual sampling and dev scores."""
import argparse
import contextlib
import io
import math
from pathlib import Path
import sys

from online_dev_config import (PROJECT, STAGE, jsonl, read, require, sha,
                               source_files, validate_manifest, verify_locks)


def close(a, b, label):
    require(math.isfinite(a) and math.isclose(a, b, rel_tol=1e-12, abs_tol=1e-12), label)


def replay_feedback(history, cadence=401, initial_lr=.0001):
    """Recompute rules directly rather than trusting the training controller."""
    best, best_step, anchor, bad, lr = None, None, None, 0, initial_lr
    state = None
    for i, row in enumerate(history, 1):
        step = row['step']
        require(step == i * cadence, 'Missing/repeated/noncadence dev evaluation')
        score = row.get('score', row.get('official', {}).get('AC_f1'))
        require(isinstance(score, (int, float)) and math.isfinite(score) and 0 <= score <= 1,
                'Nonfinite/invalid dev score')
        if 'official' in row:
            close(score, row['official']['AC_f1'], 'Feedback control score differs from official AC')
        require(state is None or not state['stop'], 'Training evaluated after early stop')
        new_best = best is None or score >= best
        if new_best:
            best, best_step = score, step
        if anchor is None or score - anchor >= .001 - 1e-12:
            anchor, bad = score, 0
        else:
            bad += 1
        stop = bad >= 6
        reduce = bad == 3 and not stop
        before = row.get('lr_before', row.get('LR_before', [None])[0])
        after = row.get('lr_after', row.get('LR_after', [None])[0])
        close(before, lr, 'LR before feedback differs')
        if reduce:
            lr *= .5
        close(after, lr, 'LR after feedback differs')
        state = dict(best_score=best, best_step=best_step, anchor_score=anchor, bad_count=bad,
                     eval_count=i, last_eval_step=step, new_best=new_best, reduce_lr=reduce, stop=stop)
        for key, value in state.items():
            require(row.get(key) == value, f'Feedback replay mismatch: {key}')
    require(state is not None, 'No online dev feedback')
    return state


def validate_dev_inputs(m, cfg, smoke):
    source_rows = jsonl(Path(m['data_dir'])/'dev.jsonl')
    source_gold = jsonl(Path(m['document_data_dir'])/'dev.jsonl')
    for name, hash_name in [('dev_file', 'dev_file_sha256'),
                            ('dev_document_file', 'dev_document_sha256')]:
        require(sha(cfg[name]) == cfg[hash_name], 'Configured dev hash changed')
    actual_rows, actual_gold = jsonl(cfg['dev_file']), jsonl(cfg['dev_document_file'])
    if smoke:
        doc_ids = list(dict.fromkeys(r['document_id'] for r in source_rows))[:2]
        expected_rows = [r for r in source_rows if r['document_id'] in doc_ids]
        expected_gold = [r for r in source_gold if r['document_id'] in doc_ids]
        require(len(expected_gold) == 2, 'Smoke requires exactly two full dev abstracts')
    else:
        require(Path(cfg['dev_file']).resolve() == Path(m['data_dir'])/'dev.jsonl',
                'Formal dev must use original frozen sentence split')
        require(Path(cfg['dev_document_file']).resolve() == Path(m['document_data_dir'])/'dev.jsonl',
                'Formal dev must use original frozen gold documents')
        require(len(source_rows) == 1593, 'Formal dev sentence coverage differs')
        expected_rows, expected_gold = source_rows, source_gold
    require(actual_rows == expected_rows and actual_gold == expected_gold,
            'Dev contents/order differ from frozen complete split')
    return actual_rows, actual_gold


def verify_adapter_exports(training, best_step, last_step):
    for label, step in [('best', best_step), ('last', last_step)]:
        for filename in ('adapter_model.safetensors', 'adapter_config.json'):
            require(sha(training/f'{label}-adapter'/filename) ==
                    sha(training/'checkpoints'/f'checkpoint-{step}'/filename),
                    f'{label} adapter differs from checkpoint')


def audit_sampling_prefix(records, ids, *, global_step, accumulation=16, max_epochs=4):
    import torch
    n = len(ids)
    require(n > 0 and len(set(ids)) == n, 'Empty/duplicated train IDs')
    updates_per_epoch = math.ceil(n / accumulation)
    require(0 < global_step <= max_epochs * updates_per_epoch, 'Training budget outside course')
    full, partial = divmod(global_step, updates_per_epoch)
    counts = [n] * full + ([min(partial * accumulation, n)] if partial else [])
    require(len(records) == sum(counts), 'Actual row consumption differs from optimizer steps')
    require([r['microbatch'] for r in records] == list(range(len(records))), 'Microbatch evidence repeats/skips')
    position = 0
    for epoch, count in enumerate(counts):
        expected = list(range(n)) if epoch == 0 else torch.randperm(
            n, generator=torch.Generator().manual_seed(42 + epoch)).tolist()
        rows = records[position:position + count]
        require(all(r['epoch'] == epoch for r in rows), 'Unexpected/nonmonotonic epoch')
        require([r['index'] for r in rows] == expected[:count], 'Sampling permutation/prefix differs')
        require(all(r['id'] == ids[r['index']] for r in rows), 'Sampling ID/index mismatch')
        # An incomplete gradient accumulation group is flushed at epoch end.
        require(all(r['optimizer_step_before'] == epoch * updates_per_epoch + j // accumulation
                    for j, r in enumerate(rows)), 'Sampling optimizer boundary differs')
        position += count
    return {'passed': True, 'global_step': global_step, 'observed_rows': len(records),
            'epoch_counts': counts, 'complete_epochs': full,
            'partial_epoch_rows': counts[-1] if partial else 0}


def verify_training(run_dir):
    from online_dev_trainer import validate_online_checkpoint
    run_dir = Path(run_dir)
    state = read(run_dir / 'run.json')
    m = read(state['manifest'])
    validate_manifest(m)
    require(sha(state['manifest']) == state['manifest_sha256'], 'Manifest hash differs')
    require(set(state['source_hashes']) == {str(p) for p in source_files()}, 'New source hash inventory differs')
    verify_locks(state['source_hashes'])
    for name, digest in state['source_hashes'].items():
        require(sha(run_dir / 'source_snapshot' / Path(name).relative_to(PROJECT)) == digest,
                'New source snapshot differs')
    training = run_dir / 'training'
    cfg, metrics = read(training / 'config.json'), read(training / 'train_metrics.json')
    trainer = read(training / 'trainer_state.json')
    validate_dev_inputs(m, cfg, state['smoke'])
    horizon, cadence, epochs = (4, 2, 2) if state['smoke'] else (3208, 401, 4)
    require(cfg['max_steps'] == horizon == trainer['max_steps'], 'Scheduler horizon differs')
    step = trainer['global_step']
    require(0 < step <= horizon and step % cadence == 0, 'Wrong terminal step')
    require(metrics['status'] == 'completed' and metrics['finite_loss'], 'Invalid terminal training state')
    require(step == metrics['global_step'], 'Final metric step differs')
    for key in ('r', 'alpha', 'dropout', 'targets', 'microbatch', 'gradient_accumulation',
                'max_len', 'precision', 'seed', 'model_path', 'learning_rate', 'scheduler',
                'init_adapter', 'enable_thinking'):
        require(cfg[key] == m[key], f'Recipe differs: {key}')
    feedback = jsonl(training / 'feedback.jsonl')
    replay = replay_feedback(feedback, cadence=cadence)
    require(step == replay['last_eval_step'], 'Last online eval differs')
    require(metrics['best_step'] == replay['best_step'], 'Best selection differs')
    reason = 'early_stop' if replay['stop'] else 'budget_exhausted'
    require(metrics['stop_reason'] == reason, 'Stop reason differs')
    require(replay['stop'] or step == horizon, 'Premature unexplained training stop')
    require(trainer['best_global_step'] == replay['best_step'], 'HF best overwrote controller')
    require(Path(trainer['best_model_checkpoint']) == training/'checkpoints'/f"checkpoint-{replay['best_step']}",
            'Trainer best path differs')
    verify_adapter_exports(training, replay['best_step'], step)
    ids = cfg['ordered_ids']
    sorted_ids = Path(m['ordered_ids_file']).read_text().splitlines()
    if state['smoke']:
        lengths = cfg['source_token_lengths']
        selected = sorted(sorted(range(len(lengths)), key=lambda i: (-lengths[i], i))[:32])
        require(len(lengths) == len(sorted_ids) and selected == cfg['original_sorted_indices'],
                'Smoke is not longest32 in original course order')
        require(ids == [sorted_ids[i] for i in selected], 'Smoke selected IDs differ')
    else:
        require(ids == sorted_ids, 'Frozen Sorted1 IDs differ')
    sampling = audit_sampling_prefix(jsonl(training / 'observed-order.jsonl'), ids,
                                    global_step=step, max_epochs=epochs)
    require(read(training / 'sampling-audit.json') == sampling, 'Saved sampling audit differs')
    checkpoints = list(range(cadence, step + 1, cadence))
    require(sorted(int(p.name.split('-')[-1]) for p in (training/'checkpoints').glob('checkpoint-*'))
            == checkpoints, 'Checkpoint inventory differs')
    for s, row in zip(checkpoints, feedback):
        cp = training / 'checkpoints' / f'checkpoint-{s}'
        validate_online_checkpoint(cp, s, horizon)
        from curriculum_trainer import _load_state
        scheduler = _load_state(cp, 'scheduler')
        optimizer = _load_state(cp, 'optimizer')
        require(scheduler['last_epoch'] == s, 'Scheduler stepped other than once per update')
        require(scheduler['warmup_steps'] == (1 if state['smoke'] else 97), 'Wrong warmup')
        expected_lr = row['lr_after']
        close(scheduler['plateau_scale'], expected_lr/.0001, 'Checkpoint plateau scale differs')
        require(all(x == .0001 for x in scheduler['base_lrs']), 'Checkpoint base LR differs')
        require(len(scheduler['_last_lr']) == len(optimizer['param_groups']), 'LR group count differs')
        for saved_lr, group in zip(scheduler['_last_lr'], optimizer['param_groups']):
            close(saved_lr, expected_lr, 'Checkpoint scheduler actual LR differs')
            close(group['lr'], expected_lr, 'Checkpoint optimizer actual LR differs')
        saved_feedback = read(cp / 'feedback_state.json')['controller']
        for key in ('best_step', 'best_score', 'anchor_score', 'bad_count', 'last_eval_step', 'eval_count'):
            require(saved_feedback[key] == row[key], 'Checkpoint feedback differs')
    require(read(training/'token_audit.json')['overlength_count'] == 0, 'Train truncation occurred')
    losses = [x for x in trainer['log_history'] if 'loss' in x]
    require([x['step'] for x in losses] == list(range(1, step + 1)), 'Loss trace incomplete')
    require(all(math.isfinite(x['loss']) and x['loss'] >= 0 for x in losses), 'Nonfinite training loss')
    return {'status': 'passed', 'updates': step, 'best_step': replay['best_step'],
            'stop_reason': reason, 'sampling': sampling, 'feedback_replayed': True}


def verify_dev_scores(run_dir):
    from verify_sorted_first_run import fixed_evaluation_modules
    evaluation, merge = fixed_evaluation_modules()
    state = read(Path(run_dir) / 'run.json')
    cfg = read(Path(run_dir) / 'training/config.json')
    m = read(state['manifest'])
    rows, gold = validate_dev_inputs(m, cfg, state['smoke'])
    results = []
    for record in jsonl(Path(run_dir) / 'training/feedback.jsonl'):
        paths = record['paths']
        predictions = jsonl(paths['predictions'])
        completion = read(paths['complete'])
        require(completion['status'] == 'completed' and completion['count'] == len(rows)
                and completion['output_sha256'] == sha(paths['predictions']), 'Dev completion manifest differs')
        require([r['id'] for r in predictions] == [r['id'] for r in rows], 'Incomplete dev generation')
        merged = merge.merge_sentence_rows(rows, merge.reparse_raw_predictions(predictions))
        require(merged == jsonl(paths['merged']), 'Merged dev predictions differ')
        with contextlib.redirect_stdout(io.StringIO()):
            recalculated, docs, _ = evaluation.evaluate(gold, merged)
        saved = read(paths['metrics'])
        require(set(recalculated['official']) == set(record['official']), 'Dev metric keys differ')
        for key, value in recalculated['official'].items():
            close(value, saved['official'][key], f'Saved dev {key} differs')
            close(value, record['official'][key], f'Control dev {key} differs')
        require(recalculated['scope'] == 'full', 'Dev not full defined split')
        require(docs == read(paths['scievents']), 'Official dev artifact differs')
        results.append({'step': record['step'], 'official': recalculated['official']})
    return results


def verify(run_dir, include_test=False):
    result = verify_training(run_dir)
    result['dev_scores_recomputed'] = verify_dev_scores(run_dir)
    if include_test:
        from run_sorted_first_online_dev import test_diagnostics
        state = read(Path(run_dir)/'run.json')
        require(not state['smoke'], 'No test on smoke')
        actual = test_diagnostics(read(state['manifest']), state['test_prediction'], state['test_merged'])
        require(actual == state['test_scores'], 'Test diagnostic scores differ')
        require(len(jsonl(state['test_prediction'])) == 1627, 'Test coverage incomplete')
        require(sha(state['best_checkpoint'] + '/adapter_model.safetensors') ==
                sha(Path(run_dir)/'training/best-adapter/adapter_model.safetensors'), 'Final adapter is not dev best')
        result['test_scores_recomputed'] = True
    return result


if __name__ == '__main__':
    import json
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--include-test', action='store_true')
    args = p.parse_args()
    print(json.dumps(verify(args.output, args.include_test), indent=2))
