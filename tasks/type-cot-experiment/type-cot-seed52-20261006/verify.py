"""Independent two-stage checks: real sampling, safe states, dense feedback and dev scores."""
import contextlib
import io
import math
from pathlib import Path
import sys

from config import PROJECT,EVAL_STEPS,read,jsonl,save,sha,require,validate_manifest,verify_locks,source_files
from verify_sorted_first_online_dev import (close,validate_dev_inputs,verify_adapter_exports)
from sampling import audit_sampling_prefix


def replay_feedback(rows,schedule):
    best=best_step=anchor=None
    bad=0;lr=.0001;state=None
    require(rows and [r['step'] for r in rows]==schedule[:len(rows)],'Feedback schedule differs')
    for i,row in enumerate(rows,1):
        require(state is None or not state['stop'],'Evaluated after stop')
        step,score=row['step'],row['score']
        require(math.isfinite(score) and 0<=score<=1,'Invalid dev score')
        close(score,row['official']['AC_f1'],'Control score not official AC')
        new=best is None or score>=best
        if new:best,best_step=score,step
        if anchor is None or score-anchor>=.001-1e-12:anchor,bad=score,0
        else:bad+=1
        stop=bad>=6;reduce=bad==3 and not stop
        close(row['lr_before'],lr,'Wrong feedback initial LR')
        if reduce:lr*=.5
        close(row['lr_after'],lr,'Wrong feedback final LR')
        state=dict(best_score=best,best_step=best_step,anchor_score=anchor,bad_count=bad,
                   eval_count=i,last_eval_step=step,new_best=new,reduce_lr=reduce,stop=stop)
        for key,value in state.items():require(row[key]==value,f'Feedback replay differs: {key}')
    return state


def validate_saved_lr(scheduler,optimizer,row,step,warmup):
    require(scheduler['last_epoch']==step and scheduler['warmup_steps']==warmup,'Saved scheduler horizon differs')
    groups=optimizer['param_groups']
    require(len(scheduler['_last_lr'])==len(groups)==len(scheduler['base_lrs']) and groups,
            'Saved LR group count differs')
    require(all(lr==.0001 for lr in scheduler['base_lrs']),'Saved base LR differs')
    close(scheduler['plateau_scale'],row['lr_after']/.0001,'Saved LR multiplier differs')
    for lr,group in zip(scheduler['_last_lr'],groups):
        close(lr,row['lr_after'],'Saved scheduler LR differs')
        close(group['lr'],row['lr_after'],'Saved optimizer LR differs')


def validate_test_metadata(run_dir,run,tree,best_step):
    selected=run_dir/'tree/checkpoints'/f'checkpoint-{best_step}'
    require(Path(run['best_checkpoint']).resolve()==selected.resolve(),'Test must use dev-selected checkpoint')
    require(sha(selected/'adapter_model.safetensors')==sha(run_dir/'tree/best-adapter/adapter_model.safetensors'),
            'Test adapter differs from dev best')
    source=jsonl(Path(tree['data_dir'])/'test.jsonl')
    prediction=Path(run['test_prediction']);rows=jsonl(prediction)
    require(len(rows)==len(source)==1627 and [r['id'] for r in rows]==[r['id'] for r in source],
            'Test generation coverage/order differs')
    completion=read(str(prediction)+'.complete.json')
    require(completion['status']=='completed' and completion['count']==1627
            and completion['output_sha256']==sha(prediction),'Test generation completion/hash differs')
    return source,rows


def verify_cot(directory,m,smoke):
    import torch
    from curriculum_trainer import validate_resume_checkpoint,_load_state
    cfg=read(directory/'config.json');metrics=read(directory/'train_metrics.json')
    state=read(directory/'trainer_state.json');records=jsonl(directory/'observed-order.jsonl')
    steps=2 if smoke else 802
    require(cfg['initialization']=='fresh base + new LoRA' and cfg['init_adapter'] is None,'CoT not fresh')
    require(cfg['max_steps']==steps==metrics['global_step']==state['global_step'],'CoT steps differ')
    require(cfg['scheduler']=='cosine' and cfg['warmup_steps']==(1 if smoke else 25),'CoT LR recipe differs')
    require(metrics['status']=='completed' and metrics['finite_loss'],'CoT not complete')
    source=jsonl(Path(m['cot_data_dir'])/'train.jsonl')
    indices=cfg['source_indices']
    if smoke:
        lengths=cfg['source_token_lengths']
        require(len(lengths)==len(source),'CoT smoke source lengths differ')
        require(indices==sorted(sorted(range(len(source)),key=lambda i:(-lengths[i],i))[:32]),'CoT smoke not longest32')
    else:require(indices==list(range(len(source))),'CoT not fulltrain')
    ids=[source[i]['id'] for i in indices]
    require(ids==cfg['ordered_ids'],'CoT train IDs differ')
    order=torch.randperm(len(ids),generator=torch.Generator().manual_seed(m['seed'])).tolist()
    require(len(records)==len(ids) and [r['index'] for r in records]==order,'CoT shuffled coverage differs')
    require(all(r['epoch']==0 and r['microbatch']==j and r['optimizer_step_before']==j//16
                and r['id']==ids[r['index']] for j,r in enumerate(records)),'CoT actual sample boundaries differ')
    cp=directory/'checkpoints'/f'checkpoint-{steps}'
    validate_resume_checkpoint(cp,steps,steps)
    scheduler=_load_state(cp,'scheduler')
    require(scheduler['last_epoch']==steps,'CoT scheduler updates differ')
    for name in ('adapter_model.safetensors','adapter_config.json'):
        require(sha(directory/'final-adapter'/name)==sha(cp/name),'CoT final weights differ')
    losses=[x for x in state['log_history'] if 'loss' in x]
    require([x['step'] for x in losses]==list(range(1,steps+1)),'CoT loss history incomplete')
    require(all(math.isfinite(x['loss']) for x in losses),'Nonfinite CoT loss')
    require(read(directory/'token_audit.json')['overlength_count']==0,'CoT truncated')
    require(read(directory/'initialization.json')['optimizer_state_entries']==0,'CoT optimizer not new')
    return dict(passed=True,updates=steps,rows=len(ids),shuffled_complete=True)


def verify(run_dir,include_test=False):
    from curriculum_trainer import _load_state
    from online_dev_trainer import validate_online_checkpoint
    from verify_sorted_first_run import fixed_evaluation_modules
    run_dir=Path(run_dir);run=read(run_dir/'run.json');m=read(run['manifest'])
    validate_manifest(m);require(sha(run['manifest'])==run['manifest_sha256'],'Manifest changed')
    require(set(run['source_hashes'])=={str(p) for p in source_files()},'Source inventory differs')
    verify_locks(run['source_hashes'])
    for path,digest in run['source_hashes'].items():
        require(sha(run_dir/'source_snapshot'/Path(path).relative_to(PROJECT))==digest,'Source snapshot differs')
    cot=verify_cot(run_dir/'cot',m,run['smoke'])
    training=run_dir/'tree';cfg=read(training/'config.json');metrics=read(training/'train_metrics.json')
    trainer=read(training/'trainer_state.json')
    horizon,epochs=(4,2) if run['smoke'] else (3208,4)
    schedule=[2,4] if run['smoke'] else EVAL_STEPS
    history=jsonl(training/'feedback.jsonl');replay=replay_feedback(history,schedule)
    step=metrics['global_step'];require(step==trainer['global_step']==history[-1]['step'],'Tree terminal step differs')
    require(cfg['max_steps']==horizon==trainer['max_steps'],'Tree horizon differs')
    require(cfg['checkpoint_steps']==schedule,'Tree configured schedule differs')
    require(metrics['status']=='completed' and metrics['finite_loss'],'Tree not complete')
    require(metrics['best_step']==replay['best_step'],'Tree best selection differs')
    require(replay['stop'] or step==horizon,'Tree stopped prematurely')
    require(metrics['stop_reason']==('early_stop' if replay['stop'] else 'budget_exhausted'),'Stop reason differs')
    source=run_dir/'cot/final-adapter'
    require(Path(cfg['init_adapter']).resolve()==source.resolve(),'Tree adapter is not this run CoT')
    handoff=read(training/'handoff_evidence.json')
    require(handoff['model_exact'] and handoff['adapter_tensor_count']>0, 'CoT weights not loaded exactly')
    require(handoff['optimizer_state_empty'],'Tree reused optimizer state')
    require(handoff.get('scheduler_last_epoch')==0,'Tree reused scheduler')
    require(handoff['feedback']['eval_count']==0 and handoff['feedback']['best_step'] is None,'Tree reused feedback')
    require(handoff['global_step']==0 and handoff['resume_from_checkpoint'] is None,'Tree resumed old state')
    require(handoff['scheduler_plateau_scale']==1.,'Tree reused LR scale')
    require(Path(handoff['source_adapter']).resolve()==source.resolve(),'Handoff adapter differs')
    expected_files={str(p.resolve()) for p in [run_dir/'cot/config.json',run_dir/'cot/train_metrics.json',
                   source/'adapter_config.json',source/'adapter_model.safetensors']}
    require(set(handoff['source_sha256'])==expected_files,'Handoff source inventory differs')
    verify_locks(handoff['source_sha256'])
    tree=m['tree_reference']
    for key in ('r','alpha','dropout','targets','microbatch','gradient_accumulation','max_len','precision','model_path','learning_rate','scheduler','enable_thinking'):
        require(cfg[key]==tree[key],f'Tree recipe differs: {key}')
    require(cfg['seed']==m['seed']==run['seed']==52 and read(run_dir/'cot/config.json')['seed']==52,'Actual seed differs')
    require(run['cot_dev_diagnostic'] is False and not (run_dir/'cot-dev').exists(),'CoT dev unexpectedly executed')
    dev_rows,dev_gold=validate_dev_inputs(tree,cfg,run['smoke'])
    ids=cfg['ordered_ids'];full_ids=Path(tree['ordered_ids_file']).read_text().splitlines()
    if run['smoke']:
        lengths=cfg['source_token_lengths'];selected=sorted(sorted(range(len(full_ids)),key=lambda i:(-lengths[i],i))[:32])
        require(cfg['original_sorted_indices']==selected and ids==[full_ids[i] for i in selected],'Tree smoke sampling differs')
    else:require(ids==full_ids,'Tree sorted IDs differ')
    sampling=audit_sampling_prefix(jsonl(training/'observed-order.jsonl'),ids,global_step=step,max_epochs=epochs,seed=m['seed'])
    require(sampling==read(training/'sampling-audit.json'),'Saved tree sampling differs')
    verify_adapter_exports(training,replay['best_step'],step)
    checkpoints=[r['step'] for r in history]
    require(sorted(int(p.name.split('-')[-1]) for p in (training/'checkpoints').glob('checkpoint-*'))==checkpoints,'Checkpoint inventory differs')
    evaluation,merge=fixed_evaluation_modules();scored=[]
    for record in history:
        s=record['step'];cp=training/'checkpoints'/f'checkpoint-{s}'
        validate_online_checkpoint(cp,s,horizon)
        saved_feedback=read(cp/'feedback_state.json')['controller']
        for key,value in saved_feedback.items():require(value==record[key],'Saved feedback differs')
        optimizer=_load_state(cp,'optimizer');scheduler=_load_state(cp,'scheduler')
        validate_saved_lr(scheduler,optimizer,record,s,1 if run['smoke'] else 97)
        paths=record['paths'];pred=jsonl(paths['predictions']);completion=read(paths['complete'])
        require([r['id'] for r in pred]==[r['id'] for r in dev_rows],'Dev coverage/order differs')
        require(completion['status']=='completed' and completion['count']==len(dev_rows)
                and completion['output_sha256']==sha(paths['predictions']),'Dev generation completion differs')
        merged=merge.merge_sentence_rows(dev_rows,merge.reparse_raw_predictions(pred))
        require(merged==jsonl(paths['merged']),'Dev merge differs')
        with contextlib.redirect_stdout(io.StringIO()):actual,docs,_=evaluation.evaluate(dev_gold,merged)
        saved=read(paths['metrics'])
        require(len(actual['official'])==18 and actual['scope']=='full','Not full official dev metrics')
        for key,value in actual['official'].items():
            close(value,record['official'][key],'Dev control metric differs');close(value,saved['official'][key],'Saved dev metric differs')
        require(docs==read(paths['scievents']),'Dev official artifact differs')
        scored.append(dict(step=s,AC_f1=actual['official']['AC_f1']))
    require(trainer['best_global_step']==replay['best_step'],'Trainer best differs')
    losses=[r for r in trainer['log_history'] if 'loss' in r]
    require([r['step'] for r in losses]==list(range(1,step+1)) and all(math.isfinite(r['loss']) for r in losses),'Tree loss trace incomplete')
    require(read(training/'token_audit.json')['overlength_count']==0,'Tree truncated')
    result=dict(status='passed',cot=cot,tree_updates=step,best_step=replay['best_step'],sampling=sampling,
                feedback_replayed=True,dev_scores_recomputed=scored)
    if include_test:
        from run_sorted_first_online_dev import test_diagnostics
        require(not run['smoke'],'Smoke must not evaluate test')
        source,predictions=validate_test_metadata(run_dir,run,tree,replay['best_step'])
        merged=merge.merge_sentence_rows(source,merge.reparse_raw_predictions(predictions))
        require(merged==jsonl(run['test_merged']),'Raw test to offset merge differs')
        gold=jsonl(Path(tree['document_data_dir'])/'test.jsonl')
        with contextlib.redirect_stdout(io.StringIO()):test_actual,_,_=evaluation.evaluate(gold,merged)
        artifact=read(str(run['test_prediction']).removesuffix('.predictions.jsonl')+'.metrics.json')
        for key,value in test_actual['official'].items():
            close(value,run['test_official'][key],'Final official test metric differs')
            close(value,artifact['official'][key],'Test metric artifact differs')
        require(test_diagnostics(tree,run['test_prediction'],run['test_merged'])==run['test_scores'],'Test scores differ')
        result['test_scores_recomputed']=True
    return result


if __name__=='__main__':
    import argparse,json
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--include-test',action='store_true')
    args=p.parse_args();print(json.dumps(verify(args.run,args.include_test),indent=2))
