#!/usr/bin/env python3
"""Read-only independent gates for sorted-first training and full evaluation."""
import argparse
import contextlib
import io
import importlib.util
import json
import math
from pathlib import Path
import sys

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
from run_sorted_first import read,sha,jsonl,require,verify_locks,validate_manifest,select_checkpoint,score_token,PROJECT,FIXED

def fixed_evaluation_modules():
    """Load fixed files explicitly, replacing stale same-name imports and dependencies."""
    modules={}
    for name in ('ontology','linearize','align_events','evaluate','merge_sentence_predictions'):
        path=FIXED/f'{name}.py'
        spec=importlib.util.spec_from_file_location(name,path)
        module=importlib.util.module_from_spec(spec)
        sys.modules[name]=module
        spec.loader.exec_module(module)
        require(Path(module.__file__).resolve()==path.resolve(),'Incorrect fixed module origin: '+name)
        modules[name]=module
    return modules['evaluate'],modules['merge_sentence_predictions']

def compare_scores(actual,expected,label):
    require(actual.keys()==expected.keys(),f'{label}: metric keys differ')
    for key,value in actual.items():
        require(math.isfinite(value) and math.isclose(value,expected[key],rel_tol=1e-12,abs_tol=1e-12),f'{label}: {key} differs')
def coverage(rows,reference,label):
    ids=[r['id'] for r in rows];expected=[r['id'] for r in reference]
    require(len(ids)==len(expected)==len(set(ids))==len(set(expected)) and set(ids)==set(expected),f'{label}: ID coverage mismatch')
def validate_observed_order(records,ids,expected_epochs):
    require(len({*ids})==len(ids),'Repeated training IDs')
    require([r['epoch'] for r in records]==sorted(r['epoch'] for r in records),'Epoch order is nonmonotonic')
    require({r['epoch'] for r in records}==set(range(len(expected_epochs))),'Unexpected or missing epochs')
    for epoch,expected in enumerate(expected_epochs):
        observed=[r for r in records if r['epoch']==epoch]
        require([r['index'] for r in observed]==expected,f'Epoch{epoch}: actual order differs')
        require(all(r['id']==ids[r['index']] for r in observed),f'Epoch{epoch}: ID/index mismatch')
        require(sorted(expected)==list(range(len(ids))),f'Epoch{epoch}: incomplete permutation')
        if epoch: require(expected!=list(range(len(ids))),f'Epoch{epoch}: later pass sequential')

def finite(value,label):
    if isinstance(value,dict):
        for key,item in value.items(): finite(item,label+'.'+key)
    elif isinstance(value,list):
        for i,item in enumerate(value): finite(item,f'{label}[{i}]')
    elif isinstance(value,float): require(math.isfinite(value),f'{label}: nonfinite value')

def verify_training(run_dir,state,manifest):
    import torch
    from curriculum_trainer import validate_resume_checkpoint
    from sorted_first_sampler import sampling_audit
    run_dir=Path(run_dir);directory=run_dir/'training';smoke=state['smoke']
    total=4 if smoke else 4010;epochs=2 if smoke else 5
    metrics=read(directory/'train_metrics.json');cfg=read(directory/'config.json');trainer=read(directory/'trainer_state.json')
    finite(metrics,'metrics');finite(trainer.get('log_history',[]),'loss history')
    require(metrics['status']=='completed','Training incomplete')
    require(metrics['global_step']==trainer['global_step']==total,'Global update budget differs')
    require(cfg['max_steps']==trainer['max_steps']==state['scheduler_horizon']==total,'Scheduler horizon differs')
    require(metrics['finite_loss'] is True,'Finite loss guard not established')
    require(cfg['smoke']==smoke,'Smoke configuration differs')
    require(Path(cfg['data_path']).resolve()==Path(manifest['sorted_train_file']).resolve(),'Incorrect train data')
    baseline_cfg=read(Path(manifest['baseline_run_file']).parent/'training/config.json')
    for key in ('r','alpha','dropout','targets','learning_rate','scheduler','warmup_ratio','microbatch',
                'gradient_accumulation','max_len','precision','quantization','loss','seed','model_path'):
        require(cfg[key]==baseline_cfg[key],f'Baseline training recipe differs: {key}')
    require(cfg['init_adapter'] is None and cfg['enable_thinking'] is False,'Incorrect initialization/thinking')
    sorted_rows=jsonl(manifest['sorted_train_file']);source_ids=[r['id'] for r in sorted_rows]
    require(source_ids==Path(manifest['ordered_ids_file']).read_text().splitlines(),'Frozen ordered IDs differ')
    source_bytes={json.loads(line)['id']:line for line in Path(manifest['source_train_file']).read_bytes().splitlines(keepends=True)}
    require(len(source_bytes)==len(sorted_rows),'Training source duplicates/count differs')
    require(all(source_bytes[json.loads(line)['id']]==line for line in Path(manifest['sorted_train_file']).read_bytes().splitlines(keepends=True)),
            'Sorted rows payload bytes differ')
    ids=cfg['ordered_ids']
    if smoke:
        # Independent recomputation of selected longest rows; tokenized lengths are frozen in token audit.
        original_indices=cfg['original_sorted_indices']
        require(len(ids)==32 and len(original_indices)==32 and original_indices==sorted(original_indices),'Smoke selection order differs')
        require(ids==[source_ids[i] for i in original_indices],'Smoke IDs differ from sorted source')
        lengths=cfg['source_token_lengths']
        require(len(lengths)==len(source_ids) and all(isinstance(n,int) and n>0 for n in lengths),'Invalid source token lengths')
        longest=sorted(sorted(range(len(lengths)),key=lambda i:(-lengths[i],i))[:32])
        require(original_indices==longest,'Smoke subset is not longest32')
    else: require(ids==source_ids and cfg['original_sorted_indices']==list(range(len(ids))),'Full train IDs differ')
    expected=[list(range(len(ids)))]+[torch.randperm(len(ids),generator=torch.Generator().manual_seed(42+epoch)).tolist() for epoch in range(1,epochs)]
    records=jsonl(directory/'observed-order.jsonl');validate_observed_order(records,ids,expected)
    audit=read(directory/'sampling-audit.json')
    recalculated=sampling_audit(records,ids,seed=42,expected_epochs=epochs,global_step=trainer['global_step'],
                              total_optimizer_steps=total,collated_microbatches=audit['collated_microbatches'])
    require(audit==recalculated and audit['complete'] is True,'Sampling audit inconsistent/incomplete')
    steps=[2,4] if smoke else [802,1604,2406,3208,4010]
    checkpoints=sorted((directory/'checkpoints').glob('checkpoint-*'),key=lambda p:int(p.name.split('-')[-1]))
    require([int(p.name.split('-')[-1]) for p in checkpoints]==steps==state['checkpoint_steps'],'Checkpoint inventory differs')
    for checkpoint in checkpoints:
        validate_resume_checkpoint(checkpoint,expected_step=int(checkpoint.name.split('-')[-1]),expected_horizon=total)
    losses=[x for x in trainer['log_history'] if 'loss' in x]
    require([x['step'] for x in losses]==list(range(1,total+1)),'Loss history incomplete')
    require(all(x['loss']>=0 for x in losses),'Negative train loss')
    require(read(directory/'token_audit.json')['overlength_count']==0,'Overlength training rows')
    return {'updates':total,'epochs':epochs,'observed_rows':len(records),'checkpoints':steps,'sampling_verified':True}

def verify(run_dir,allow_running=False):
    run_dir=Path(run_dir).resolve();state=read(run_dir/'run.json')
    require(state['status']=='completed' or (allow_running and state['status']=='running' and state['current_activity']=='verify'),'Run incomplete')
    manifest=read(state['manifest']);validate_manifest(manifest)
    require(sha(state['manifest'])==state['manifest_sha256'],'Manifest changed')
    require(sha(state['baseline_run'])==state['baseline_sha256'],'Baseline changed')
    require(Path(state['baseline_run']).resolve()==Path(manifest['baseline_run_file']).resolve(),'Baseline reference differs')
    require(state['experiment']==manifest['experiment']=='C-CL-Mean-AppendixNoCopy-Sorted1','Wrong experiment')
    require(state['evaluation_version']=='evaluation-v2-20261003','Incorrect evaluation version')
    require(sha(state['fixed_evaluation_manifest'])==state['fixed_evaluation_manifest_sha256'],'Fixed evaluator manifest changed')
    verify_locks({**manifest['source_hashes'],**manifest['output_hashes'],**state['source_code_hashes'],**state['evaluation_reference_hashes']})
    for path,digest in state['source_code_hashes'].items():
        require(sha(run_dir/'source_snapshot'/Path(path).relative_to(PROJECT))==digest,'Source snapshot mismatch: '+path)
    result={'status':'passed','smoke':state['smoke'],'training':verify_training(run_dir,state,manifest),'input_hashes_verified':True}
    if state['smoke']:
        require(not state.get('dev_checkpoints') and not (run_dir/'checkpoint-eval').exists(),'Smoke unexpectedly evaluates dev/test')
        return result
    evaluation,merge_module=fixed_evaluation_modules()
    evaluate,run_complex_eval=evaluation.evaluate,evaluation.run_complex_eval
    merge_sentence_rows,reparse_raw_predictions=merge_module.merge_sentence_rows,merge_module.reparse_raw_predictions
    from appendix_fixed_eval import COMPLEX_KEYS
    baseline=read(state['baseline_run']);recomputed=[]
    require([x['step'] for x in state['dev_checkpoints']]==state['checkpoint_steps'],'Dev checkpoint inventory differs')
    work=[('dev',x['step'],Path(x['metrics'])) for x in state['dev_checkpoints']]
    work += [('test',state['best_step'],run_dir/'checkpoint-eval'/f"test-checkpoint-{state['best_step']}.metrics.json")]
    for split,step,metric_path in work:
        prefix=run_dir/'checkpoint-eval'/f'{split}-checkpoint-{step}'
        pred_path=prefix.with_suffix('.predictions.jsonl');merged_path=prefix.with_suffix('.merged.jsonl')
        predictions=jsonl(pred_path);sentence_gold=jsonl(Path(manifest['data_dir'])/f'{split}.jsonl')
        documents=jsonl(merged_path);gold=jsonl(Path(baseline['document_data_dir'])/f'{split}.jsonl')
        coverage(predictions,sentence_gold,f'{split} sentences');coverage(documents,gold,f'{split} documents')
        completion=read(str(pred_path)+'.complete.json')
        require(completion['status']=='completed' and completion['count']==len(sentence_gold) and completion['output_sha256']==sha(pred_path),
                'Prediction completion/hash mismatch')
        require(documents==merge_sentence_rows(sentence_gold,reparse_raw_predictions(predictions)),f'{split}: raw/merged trees differ')
        with contextlib.redirect_stdout(io.StringIO()): calculated,output_docs,_=evaluate(gold,documents)
        saved=read(metric_path);compare_scores(calculated['official'],saved['official'],f'{split}:{step}')
        require(saved['scope']=='full' and saved['documents']==len(gold),'Metric scope differs')
        require(read(metric_path.with_suffix('.scievents.json'))==output_docs,'Official prediction artifact differs')
        if split=='test':
            require(saved.get('complex',{}).get('exit_code')==0 and COMPLEX_KEYS.issubset(saved.get('complex',{}).get('scores',{})),'Missing complex scores')
            complex_scores=run_complex_eval(metric_path.with_suffix('.scievents.json'),sys.executable)
            compare_scores(complex_scores['scores'],saved['complex']['scores'],'Complex test')
            compare_scores(calculated['official'],state['test_official'],'Run test')
            require(Path(state['test_prediction'])==pred_path and Path(state['test_merged'])==merged_path,'Test output pointers differ')
        else:
            row=next(x for x in state['dev_checkpoints'] if x['step']==step)
            require(row['checkpoint']==str(run_dir/'training/checkpoints'/f'checkpoint-{step}'),'Dev adapter pointer differs')
            compare_scores({'AC_f1':row['AC_f1']},{'AC_f1':calculated['official']['AC_f1']},'Dev selection')
        recomputed.append({'split':split,'step':step,'official':calculated['official']})
    best=select_checkpoint(state['dev_checkpoints'])
    require(best['step']==state['best_step'] and best['checkpoint']==state['best_checkpoint'] and best['metrics']==state['best_dev_metrics'],'Dev selection differs')
    verify_locks(state['test_token_source_hashes'])
    token,token_preds=score_token(Path(manifest['data_dir'])/'test.jsonl',state['test_prediction'],PROJECT/'data/SciEvents/test.json')
    require(token==state['test_token'],'Raw token diagnostics/scores differ')
    compare_scores(token['official'],state['test_token_official'],'Run raw token test')
    require(read(state['test_token_predictions'])==token_preds,'Raw token prediction artifact differs')
    report=run_dir/'sorted_first_training_report.md';require(report.is_file(),'Report missing')
    for scores in [state['test_official'],state['test_token_official']]+[read(x['metrics'])['official'] for x in state['dev_checkpoints']]:
        require(' | '.join(f'{scores[k]*100:.4f}' for k in ['TI_f1','TC_f1','AI_f1','AC_f1','EC_C_f1']) in report.read_text(),'Report scores differ')
    result.update(best_step=best['step'],recomputed_metrics=recomputed,raw_token_recomputed=True,complex_recomputed=True)
    return result

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run-dir',type=Path,required=True)
    a=p.parse_args();print(json.dumps(verify(a.run_dir),ensure_ascii=False,indent=2))
