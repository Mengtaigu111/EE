#!/usr/bin/env python3
"""One continuous sorted-first run, repaired offset selection and raw token test."""
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import shutil
import sys

HERE=Path(__file__).resolve().parent
STAGE=HERE.parent
PROJECT=STAGE.parents[1]
FIXED=STAGE/'evaluation-v2-20261003/scripts'

def read(path): return json.loads(Path(path).read_text())
def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def jsonl(path): return [json.loads(x) for x in Path(path).read_text().splitlines() if x.strip()]
def save(path,value):
    path=Path(path);tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n');tmp.replace(path)
def require(ok,message):
    if not ok: raise ValueError(message)
def verify_locks(locks):
    for path,digest in locks.items(): require(sha(path)==digest,f'Provenance changed: {path}')
def validate_manifest(m):
    expected={'seed':42,'total_optimizer_steps':4010,'effective_batch_size':16,'steps_per_epoch':802,
              'checkpoint_steps':[802,1604,2406,3208,4010]}
    for key,value in expected.items(): require(m[key]==value,f'Manifest schedule mismatch: {key}')
def select_checkpoint(results):
    require(bool(results),'No dev checkpoints')
    require(all(math.isfinite(r['AC_f1']) for r in results),'Nonfinite dev scores')
    return max(results,key=lambda r:(r['AC_f1'],r['step']))
def token_module():
    from verify_sorted_first_run import fixed_evaluation_modules
    fixed_evaluation_modules()
    path=STAGE/'token-evaluation-20261003/score_raw_token.py'
    spec=importlib.util.spec_from_file_location('sorted_first_raw_token',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module

def score_token(sentence_path,prediction_path,gold_path):
    module=token_module();gold=read(gold_path)
    docs,diag=module.merge_predictions(jsonl(sentence_path),jsonl(prediction_path))
    require(set(docs)=={d['doc_id'] for d in gold},'Token document IDs differ')
    predictions=[{'doc_id':d['doc_id'],'events':docs[d['doc_id']]} for d in gold]
    values=module.utils.calculate_F1([d['events'] for d in gold],[d['events'] for d in predictions],'token')
    return {'official':dict(zip(module.KEYS,map(float,values))),'diagnostics':diag,
            'mode':'official token, raw parser output, no alignment or deduplication'},predictions

def report(run_dir,state):
    keys=['TI_f1','TC_f1','AI_f1','AC_f1','EC_C_f1']
    lines=['# Sorted-first full-train results','','First epoch sorted; four later epochs shuffled. '
           'Original AppendixNoCopy prompt, raw Qwen3-8B, one continuous 4010-update run, seed42.','',
           '| Offset dev checkpoint | TI | TC | AI | AC | EC |','|---|---:|---:|---:|---:|---:|']
    for row in state['dev_checkpoints']:
        scores=read(row['metrics'])['official']
        lines.append('| '+str(row['step'])+' | '+' | '.join(f'{scores[k]*100:.4f}' for k in keys)+' |')
    lines += ['',f"Selected step {state['best_step']} by full dev offset AC F1; later-step tie.",'',
              '| Selected test metric | TI | TC | AI | AC | EC |','|---|---:|---:|---:|---:|---:|']
    for label,scores in [('offset',state['test_official']),('raw token',state['test_token_official'])]:
        lines.append('| '+label+' | '+' | '.join(f'{scores[k]*100:.4f}' for k in keys)+' |')
    lines += ['','Raw token uses original SciEvents gold, parsed raw predictions, no alignment or deduplication. '
              'Official shared-trigger limitations retained; single-seed exploratory comparison.','',
              'Corrected baseline test offset: '+json.dumps(read(state['baseline_corrected_test_metrics'])['official']),
              '',f"Manifest: `{state['manifest']}`; SHA256 `{state['manifest_sha256']}`.",
              f"Sampling audit: `{run_dir/'training/sampling-audit.json'}`.",'']
    (run_dir/'sorted_first_training_report.md').write_text('\n'.join(lines))

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-dir',type=Path,required=True);p.add_argument('--manifest',type=Path,required=True)
    p.add_argument('--smoke',action='store_true');p.add_argument('--batch-size',type=int,default=4)
    a=p.parse_args();require('TMUX' in os.environ,'GPU execution must run in tmux')
    sys.path.insert(0,str(STAGE/'scripts'))
    from run_c_queue import run_stage
    from appendix_fixed_eval import evaluate_checkpoint
    from run_appendix_easy2_curriculum import corrected_baseline_paths
    m=read(a.manifest);validate_manifest(m);verify_locks({**m['source_hashes'],**m['output_hashes']})
    baseline_dir=Path(m['baseline_run_file']).parent
    baseline=read(m['baseline_run_file']);cfg=read(baseline_dir/'training/config.json')
    require(read(baseline_dir/'training/trainer_state.json')['global_step']==4010,'Baseline actual budget mismatch')
    corrected=corrected_baseline_paths();fixed_manifest=STAGE/'evaluation-v2-20261003/version_manifest.json'
    total,cadence=(4,2) if a.smoke else (4010,802)
    a.run_dir=a.run_dir.resolve();a.run_dir.mkdir(parents=True,exist_ok=False)
    state={'experiment':m['experiment'],'status':'running','current_activity':'initialize','smoke':a.smoke,
        'started_at':datetime.now(timezone.utc).isoformat(),'seed':42,'manifest':str(a.manifest.resolve()),
        'manifest_sha256':sha(a.manifest),'baseline_run':str(Path(m['baseline_run_file']).resolve()),
        'baseline_sha256':sha(m['baseline_run_file']),'total_optimizer_steps':total,'scheduler_horizon':total,
        'checkpoint_steps':list(range(cadence,total+1,cadence)), 'initialization':cfg['model_path'],'init_adapter':None,
        'evaluation_version':'evaluation-v2-20261003','fixed_evaluation_manifest':str(fixed_manifest),
        'fixed_evaluation_manifest_sha256':sha(fixed_manifest),'baseline_corrected_best_step':corrected['step'],
        'baseline_corrected_dev_metrics':corrected['dev'],'baseline_corrected_test_metrics':corrected['test']}
    source_files=[HERE/name for name in ('run_sorted_first.py','verify_sorted_first_run.py','build_sorted_first_data.py',
        'train_sorted_first.py','sorted_first_sampler.py','curriculum_trainer.py','appendix_fixed_eval.py','predict_appendix_fixed.py',
        'run_appendix_easy2_curriculum.py')]
    source_files += sorted(FIXED.glob('*.py'))+[STAGE/'scripts'/n for n in ('run_c_queue.py','model_utils.py','predict.py','ontology.py')]
    source_files += [PROJECT/'utils.py',PROJECT/'eval_complex_quick_f1.py',STAGE/'token-evaluation-20261003/score_raw_token.py']
    state['source_code_hashes']={str(x):sha(x) for x in source_files}
    references=[Path(corrected['dev']),Path(corrected['test']),fixed_manifest,
                STAGE/'evaluation-v2-20261003/results/dev_selection.json',STAGE/'evaluation-v2-20261003/results/test_selection.json',
                PROJECT/'data/SciEvents/test.json']
    references += [Path(m['data_dir'])/f'{split}.jsonl' for split in ('dev','test')]
    references += [Path(baseline['document_data_dir'])/f'{split}.jsonl' for split in ('dev','test')]
    state['evaluation_reference_hashes']={str(x):sha(x) for x in references}
    locks={**m['source_hashes'],**m['output_hashes'],**state['source_code_hashes'],**state['evaluation_reference_hashes'],
           state['manifest']:state['manifest_sha256'],state['baseline_run']:state['baseline_sha256']}
    for file in source_files:
        target=a.run_dir/'source_snapshot'/file.relative_to(PROJECT);target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(file,target)
    save(a.run_dir/'run.json',state)
    try:
        verify_locks(locks);state['current_activity']='train';save(a.run_dir/'run.json',state)
        cmd=[sys.executable,str(HERE/'train_sorted_first.py'),'--data',m['sorted_train_file'],'--output',str(a.run_dir/'training'),
             '--baseline',str(baseline_dir),'--total-steps',str(total),'--save-steps',str(cadence)]
        if a.smoke: cmd+=['--smoke']
        run_stage(cmd,a.run_dir/'training.log');verify_locks(locks)
        from verify_sorted_first_run import verify_training
        verify_training(a.run_dir,state,m)
        if not a.smoke:
            args=argparse.Namespace(data_dir=Path(m['data_dir']),document_data_dir=Path(baseline['document_data_dir']),
                                    model=cfg['model_path'],seed=42,batch_size=a.batch_size)
            dev=[]
            for step in state['checkpoint_steps']:
                state['current_activity']=f'evaluate_dev_{step}';save(a.run_dir/'run.json',state);verify_locks(locks)
                cp=a.run_dir/'training/checkpoints'/f'checkpoint-{step}'
                metrics,_,_=evaluate_checkpoint(cp,'dev',args,a.run_dir,baseline['max_new_tokens'])
                dev.append({'step':step,'checkpoint':str(cp),'AC_f1':metrics['official']['AC_f1'],
                            'metrics':str(a.run_dir/'checkpoint-eval'/f'dev-checkpoint-{step}.metrics.json')})
                state['dev_checkpoints']=dev;save(a.run_dir/'run.json',state)
            best=select_checkpoint(dev);state.update(best_checkpoint=best['checkpoint'],best_step=best['step'],best_dev_metrics=best['metrics'])
            state['current_activity']='evaluate_test';save(a.run_dir/'run.json',state);verify_locks(locks)
            test,pred,merged=evaluate_checkpoint(Path(best['checkpoint']),'test',args,a.run_dir,baseline['max_new_tokens'])
            token,token_preds=score_token(args.data_dir/'test.jsonl',pred,PROJECT/'data/SciEvents/test.json')
            token_path=a.run_dir/'checkpoint-eval'/f'test-checkpoint-{best["step"]}.token.predictions.json'
            save(token_path,token_preds)
            state.update(test_official=test['official'],test_prediction=str(pred),test_merged=str(merged),test_token=token,
                         test_token_official=token['official'],test_token_predictions=str(token_path),test_token_source_hashes={
                         str(x):sha(x) for x in [args.data_dir/'test.jsonl',pred,PROJECT/'data/SciEvents/test.json',
                                                PROJECT/'utils.py',FIXED/'linearize.py',STAGE/'token-evaluation-20261003/score_raw_token.py']})
            report(a.run_dir,state)
        verify_locks(locks);state['current_activity']='verify';save(a.run_dir/'run.json',state)
        from verify_sorted_first_run import verify
        result=verify(a.run_dir,allow_running=True);save(a.run_dir/'verification.json',result)
        state.update(status='completed',current_activity='completed',finished_at=datetime.now(timezone.utc).isoformat())
        save(a.run_dir/'run.json',state)
    except BaseException as exc:
        state.update(status='failed',error=repr(exc),failed_at=datetime.now(timezone.utc).isoformat());save(a.run_dir/'run.json',state)
        raise
    print(json.dumps(state,ensure_ascii=False,indent=2))
if __name__=='__main__': main()
