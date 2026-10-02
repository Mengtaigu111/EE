#!/usr/bin/env python3
"""Read-only final verification; recompute official scores without rewriting eval files."""
import argparse
import contextlib
import hashlib
import io
import json
import math
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent/'scripts'))


def read(path):
    return json.loads(Path(path).read_text())


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def finite(value, label):
    if isinstance(value, dict):
        for key,item in value.items():
            finite(item,f'{label}.{key}')
    elif isinstance(value,list):
        for i,item in enumerate(value):
            finite(item,f'{label}[{i}]')
    elif isinstance(value,float):
        require(math.isfinite(value),f'nonfinite number: {label}')


def jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def coverage(rows, reference, label, count):
    ids = [row['id'] for row in rows]
    expected = [row['id'] for row in reference]
    require(len(ids)==count and len(set(ids))==count and set(ids)==set(expected),
            f'{label}: missing, duplicate, or unknown IDs')


def compare_scores(actual, expected, label):
    require(actual.keys()==expected.keys(),f'{label}: metric keys differ')
    for key,value in actual.items():
        require(math.isfinite(value) and math.isclose(value,expected[key],rel_tol=1e-12,abs_tol=1e-12),
                f'{label}: {key} mismatch {value} != {expected[key]}')


def verify(run_dir, allow_smoke=False):
    from curriculum_trainer import validate_resume_checkpoint
    run_dir = Path(run_dir).resolve()
    state = read(run_dir/'run.json')
    require(state['status']=='completed','run is not completed')
    smoke = state.get('smoke',False)
    require(not smoke or allow_smoke,'smoke run requires --allow-smoke')
    require(digest(state['manifest'])==state['manifest_sha256'],'run manifest changed')
    require(digest(state['baseline_run'])==state['baseline_sha256'],'baseline run changed')
    manifest = read(state['manifest'])
    for path,expected in {**manifest['source_hashes'],**manifest['output_hashes']}.items():
        require(digest(path)==expected,f'provenance changed: {path}')
    for path,expected in state.get('source_code_hashes',{}).items():
        require(digest(path)==expected,f'runtime source changed: {path}')
    baseline = read(state['baseline_run'])
    total,boundary = (4,2) if smoke else (4010,2005)
    expected_steps = [2,4] if smoke else [802,1604,2005,2406,3208,4010]
    require(state['total_optimizer_steps']==state['scheduler_horizon']==total,'total/horizon mismatch')
    require(state['stage1_optimizer_steps']==state['stage2_optimizer_steps']==boundary,'phase budget mismatch')
    require(state['checkpoint_steps']==expected_steps,'checkpoint schedule mismatch')
    require(state['init_adapter'] is None and state['initialization']==manifest['base_model'],'wrong initialization')
    require(Path(state['prompt_source']).resolve()==Path(manifest['source_train_file']).resolve(),'wrong prompt source')
    require(Path(manifest['stage2_data']).read_bytes()==Path(manifest['source_train_file']).read_bytes(),
            'Stage2 is not byte-identical to baseline')
    checkpoints = sorted([*(run_dir/'stage1/checkpoints').glob('checkpoint-*'),
                          *(run_dir/'stage2/checkpoints').glob('checkpoint-*')],
                         key=lambda p:int(p.name.split('-')[-1]))
    require([int(p.name.split('-')[-1]) for p in checkpoints]==expected_steps,'actual checkpoint inventory mismatch')
    for checkpoint in checkpoints:
        validate_resume_checkpoint(checkpoint,expected_step=int(checkpoint.name.split('-')[-1]),expected_horizon=total)
    phases = []
    for phase,start,stop in [(1,0,boundary),(2,boundary,total)]:
        directory = run_dir/f'stage{phase}'
        metrics,trainer,cfg = [read(directory/name) for name in ['train_metrics.json','trainer_state.json','config.json']]
        finite(metrics,f'stage{phase}.metrics');finite(trainer['log_history'],f'stage{phase}.history')
        require(metrics['status']=='completed','phase not completed')
        require(metrics['smoke']==cfg['smoke']==smoke,'phase smoke flag differs')
        require(metrics['global_step']==trainer['global_step']==stop,'phase final global step mismatch')
        require(metrics['phase_optimizer_steps']==stop-start and metrics['phase_start_step']==start,'phase update mismatch')
        require(metrics['scheduler_horizon']==trainer['max_steps']==cfg['max_steps']==cfg['scheduler_horizon']==total,
                'phase scheduler horizon mismatch')
        require(cfg['phase_start_step']==start and cfg['phase_stop_step']==stop,'phase config boundary mismatch')
        require(cfg['init_adapter'] is None and cfg['enable_thinking'] is False,'initialization/thinking changed')
        for key in ['r','alpha','dropout','targets','learning_rate','scheduler','warmup_ratio','microbatch',
                    'gradient_accumulation','max_len','precision','quantization','loss','seed','model_path']:
            require(cfg[key]==manifest['baseline_training_config'][key],f'baseline recipe mismatch: {key}')
        require(cfg['data_path']==manifest[f'stage{phase}_data'],'wrong phase dataset')
        require(read(directory/'token_audit.json')['overlength_count']==0,'overlength training examples')
        losses = [entry for entry in trainer['log_history'] if start < entry.get('step',0)<=stop and 'loss' in entry]
        require([entry['step'] for entry in losses]==list(range(start+1,stop+1)),'missing phase loss history')
        require(all(entry['loss']>=0 for entry in losses),'negative logged loss')
        require(metrics['phase_train_loss']>=0 and metrics['first_logged_loss']==losses[0]['loss']
                and metrics['last_logged_loss']==losses[-1]['loss'],'phase loss summary mismatch')
        require(digest(directory/'adapter/adapter_model.safetensors')==digest(Path(metrics['checkpoint'])/'adapter_model.safetensors'),
                'final phase adapter differs from checkpoint')
        phases.append({'phase':phase,'updates':stop-start,'global_step':stop,'phase_train_loss':metrics['phase_train_loss']})
    evidence = read(run_dir/'stage2/resume_evidence.json')
    require(evidence==state['resume_evidence'],'run resume evidence differs from phase evidence')
    for key in ['model_exact','optimizer_exact','scheduler_exact','rng_exact']:
        require(evidence[key] is True,f'resume equality failed: {key}')
    require(evidence['global_step']==evidence['expected_step']==evidence['scheduler_last_epoch']==boundary,
            'resume step or scheduler position mismatch')
    require(evidence['horizon']==total and evidence['data_skip'] is False,'resume horizon/data skip mismatch')
    require(Path(evidence['checkpoint']).resolve()==(run_dir/f'stage1/checkpoints/checkpoint-{boundary}').resolve(),
            'wrong phase boundary checkpoint resumed')
    finite(evidence,'resume')
    result = {'status':'passed','run_dir':str(run_dir),'smoke':smoke,'total_updates':total,
              'phase_updates':[boundary,total-boundary],'checkpoint_steps':expected_steps,
              'checkpoint_integrity_verified':True,'resume_equality_verified':True,
              'input_hashes_verified':True,'phases':phases}
    if smoke:
        require(not state.get('dev_checkpoints') and not (run_dir/'checkpoint-eval').exists(),
                'smoke unexpectedly evaluated full dev/test')
        result['evaluation']='not run (smoke)'
        return result
    from evaluate import evaluate
    data_dir = Path(baseline['data_dir']);document_dir = Path(baseline['document_data_dir'])
    counts = {'dev':(1593,251),'test':(1627,250)}
    recomputed = []
    for split,step,metric_path in [('dev',item['step'],Path(item['metrics'])) for item in state['dev_checkpoints']]+[
            ('test',state['best_step'],run_dir/'checkpoint-eval'/f"test-checkpoint-{state['best_step']}.metrics.json")]:
        prefix = run_dir/'checkpoint-eval'/f'{split}-checkpoint-{step}'
        predicted = prefix.with_suffix('.predictions.jsonl');merged = prefix.with_suffix('.merged.jsonl')
        sentence_predictions = jsonl(predicted);document_predictions = jsonl(merged)
        coverage(sentence_predictions,jsonl(data_dir/f'{split}.jsonl'),f'{split} sentences',counts[split][0])
        gold = jsonl(document_dir/f'{split}.jsonl')
        coverage(document_predictions,gold,f'{split} documents',counts[split][1])
        completion = read(str(predicted)+'.complete.json')
        require(completion['status']=='completed' and completion['count']==counts[split][0], 'prediction completion mismatch')
        require(completion['output_sha256']==digest(predicted),'prediction file hash mismatch')
        with contextlib.redirect_stdout(io.StringIO()):
            calculated,_,_ = evaluate(gold,document_predictions)
        saved = read(metric_path)
        compare_scores(calculated['official'],saved['official'],f'{split} checkpoint{step}')
        require(saved['scope']=='full' and saved['documents']==counts[split][1],'stored metrics subset/count mismatch')
        if split=='test':
            compare_scores(calculated['official'],state['test_official'],'run test metrics')
            require(Path(state['test_prediction']).resolve()==predicted.resolve() and Path(state['test_merged']).resolve()==merged.resolve(),
                    'run test output paths differ')
        else:
            item = next(x for x in state['dev_checkpoints'] if x['step']==step)
            require(math.isclose(item['AC_f1'],calculated['official']['AC_f1'],abs_tol=1e-12), 'dev selection score mismatch')
        recomputed.append({'split':split,'step':step,'official':calculated['official']})
    require([x['step'] for x in state['dev_checkpoints']]==expected_steps,'dev checkpoint inventory mismatch')
    best = max(state['dev_checkpoints'],key=lambda x:(x['AC_f1'],x['step']))
    require(best['step']==state['best_step'] and best['checkpoint']==state['best_checkpoint']
            and best['metrics']==state['best_dev_metrics'],'bestdev selection mismatch')
    require(Path(state['final_dev_metrics'])==Path(state['dev_checkpoints'][-1]['metrics']), 'final dev pointer mismatch')
    require(Path(state['stage1_dev_metrics'])==run_dir/'checkpoint-eval'/f'dev-checkpoint-{boundary}.metrics.json',
            'Stage1 dev pointer mismatch')
    report = run_dir/'mean_curriculum_training_report.md'
    require(report.is_file() and report.stat().st_size>0,'training report missing')
    canonical = HERE/'mean_curriculum_training_report.md'
    require(canonical.is_file() and canonical.read_bytes()==report.read_bytes(), 'canonical report differs')
    report_text = report.read_text()
    refs = [read(next(x['metrics'] for x in baseline['dev_checkpoints'] if x['checkpoint']==baseline['best_checkpoint']))['official'],
        read(state['stage1_dev_metrics'])['official'],read(state['final_dev_metrics'])['official'],read(state['best_dev_metrics'])['official']]
    for metrics in refs:
        row = ' | '.join(f'{metrics[key]*100:.4f}' for key in ('TI_f1','TC_f1','AI_f1','AC_f1','EC_C_f1'))
        require(row in report_text, 'report metric row missing or incorrect')
    result.update(recomputed_metrics=recomputed,best_step=best['step'],report=str(report),report_sha256=digest(report))
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir',type=Path,required=True)
    parser.add_argument('--allow-smoke',action='store_true')
    args=parser.parse_args()
    print(json.dumps(verify(args.run_dir,args.allow_smoke),indent=2,ensure_ascii=False))
