"""Run fresh seeded CoT then dense-dev tree training, then test dev-best."""
import argparse
from datetime import datetime,timezone
import json
import os
from pathlib import Path
import shutil
import sys
import time
from types import SimpleNamespace

from config import HERE,PROJECT,CURRICULUM,read,jsonl,save,sha,require,build_manifest,validate_manifest,verify_locks,source_files
sys.path.insert(0,str(PROJECT/'tasks/stage2-event-tree/scripts'))
from run_c_queue import run_stage


def smoke_reload(run_dir,m):
    cfg=read(run_dir/'tree/config.json')
    out=run_dir/'smoke-reload/predictions.jsonl'
    run_stage([sys.executable,str(CURRICULUM/'predict_appendix_fixed.py'),'--data',cfg['dev_file'],
               '--output',str(out),'--model',m['model_path'],'--adapter',str(run_dir/'tree/checkpoints/checkpoint-4'),
               '--batch-size','4','--seed',str(m['seed']),'--max-new-tokens','319'],run_dir/'smoke-reload/predict.log')
    resident=jsonl(jsonl(run_dir/'tree/feedback.jsonl')[-1]['paths']['predictions'])
    loaded=jsonl(out)
    require([r['id'] for r in resident]==[r['id'] for r in loaded],'Reload IDs differ')
    require([r['raw_output'] for r in resident]==[r['raw_output'] for r in loaded],'Reload generated output differs')
    completion=read(str(out)+'.complete.json')
    require(completion['count']==len(resident) and completion['output_sha256']==sha(out),'Reload completion differs')
    result=dict(status='passed',count=len(loaded),raw_outputs_exact=True)
    save(run_dir/'smoke-reload/verification.json',result)
    return result


def test_diagnostics(tree,prediction,merged):
    """Original official token/offset plus independent text presence; test never selects."""
    import importlib.util
    from run_sorted_first_online_dev import test_diagnostics as original_diagnostics
    result=original_diagnostics(tree,prediction,merged)
    path=PROJECT/'scripts/score_tree_presence.py'
    spec=importlib.util.spec_from_file_location('mixed_presence_score',path)
    presence=importlib.util.module_from_spec(spec);spec.loader.exec_module(presence)
    result['text_presence']=presence.score(read(PROJECT/'data/SciEvents/test.json'),
        jsonl(Path(tree['data_dir'])/'test.jsonl'),jsonl(prediction))
    return result



def report(run_dir,state):
    cot=read(run_dir/'cot/train_metrics.json');tree=read(run_dir/'tree/train_metrics.json')
    rows=jsonl(run_dir/'tree/feedback.jsonl')
    lines=['# Type-CoT1 → Sorted1 dense online dev','',
           'Fresh original Qwen3-8B/new LoRA; user-final CoT prompt, one full shuffled epoch. '
           'Only this run CoT weights transfer to a new tree optimizer/scheduler/controller. '
           'Corrected gold-order AppendixNoCopy data/course; denser full-dev checks; official offset dev AC controls training.','',
           f"Status: {state['status']}; CoT updates {cot['global_step']}; tree updates {tree['global_step']}; "
           f"best tree step {tree['best_step']}; stop {tree['stop_reason']}; smoke {state['smoke']}.",'',
           '| tree step | dev TI | TC | AI | AC | EC_I | EC_C | LR after | bad count |',
           '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    keys=['TI_f1','TC_f1','AI_f1','AC_f1','EC_I_f1','EC_C_f1']
    for r in rows:
        lines.append('| '+str(r['step'])+' | '+' | '.join(f"{r['official'][k]*100:.4f}" for k in keys)+
                     f" | {r['lr_after']:.8g} | {r['bad_count']} |")
    if not state['smoke']:
        lines+=['','| test mode | TI | TC | AI | AC | EC_I | EC_C |','|---|---:|---:|---:|---:|---:|---:|']
        for mode in ('official_offset','official_raw_token','instance_offset','instance_raw_token','text_presence'):
            scores=state['test_scores'][mode]
            if mode.startswith('instance') or mode=='text_presence':scores=scores['metrics']
            lines.append('| '+mode+' | '+' | '.join(f'{scores[k]*100:.4f}' for k in keys)+' |')
    lines+=['',f"CoT training seconds {cot['wall_seconds']:.2f}; tree training+dev seconds {tree['wall_seconds']:.2f}.",
            f"Seed {state['seed']}; CoT dev diagnostic omitted. Test never controls training.", '']
    (run_dir/'report.md').write_text('\n'.join(lines))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--smoke',action='store_true')
    args=p.parse_args()
    require('TMUX' in os.environ,'GPU pipeline must run in tmux')
    m=read(args.manifest);validate_manifest(m)
    run_dir=args.output.resolve();run_dir.mkdir(parents=True,exist_ok=False)
    hashes={str(path):sha(path) for path in source_files()}
    for path in hashes:
        snapshot=run_dir/'source_snapshot'/Path(path).relative_to(PROJECT)
        snapshot.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(path,snapshot)
    state=dict(experiment=m['experiment'],status='running',current_activity='initialize',smoke=args.smoke,
               started_at=datetime.now(timezone.utc).isoformat(),manifest=str(args.manifest.resolve()),
               manifest_sha256=sha(args.manifest),source_hashes=hashes,test_controls_training=False,
               cot_steps=2 if args.smoke else 802,tree_max_steps=4 if args.smoke else 3208,
               tree_eval_steps=[2,4] if args.smoke else m['tree_eval_steps'],cot_init_adapter=None,seed=m['seed'],cot_dev_diagnostic=False)
    save(run_dir/'run.json',state)
    started=time.monotonic()
    try:
        state['current_activity']='cot_train';save(run_dir/'run.json',state)
        run_stage([sys.executable,str(HERE/'train_cot.py'),'--manifest',str(args.manifest),
                   '--output',str(run_dir/'cot')]+(['--smoke'] if args.smoke else []),run_dir/'cot-train.log')
        verify_locks({**m['frozen_hashes'],**hashes})
        from verify import verify_cot,verify
        save(run_dir/'cot-verification.json',verify_cot(run_dir/'cot',m,args.smoke))
        state['current_activity']='tree_train_online_dev';save(run_dir/'run.json',state)
        run_stage([sys.executable,str(HERE/'train_tree.py'),'--manifest',str(args.manifest),
                   '--output',str(run_dir/'tree'),'--init-adapter',str(run_dir/'cot/final-adapter')]+
                   (['--smoke'] if args.smoke else []),run_dir/'tree-train.log')
        verify_locks({**m['frozen_hashes'],**hashes})
        state['current_activity']='verify_tree_and_dev';save(run_dir/'run.json',state)
        save(run_dir/'training-verification.json',verify(run_dir))
        terminal=read(run_dir/'tree/train_metrics.json')
        state.update(best_step=terminal['best_step'],best_dev_score=terminal['best_score'],
                     last_step=terminal['global_step'],stop_reason=terminal['stop_reason'],
                     best_checkpoint=str(run_dir/'tree/checkpoints'/f"checkpoint-{terminal['best_step']}"))
        if args.smoke:
            state['current_activity']='smoke_reload';save(run_dir/'run.json',state)
            state['smoke_reload']=smoke_reload(run_dir,m)
        else:
            from appendix_fixed_eval import evaluate_checkpoint
            tree=m['tree_reference'];state['current_activity']='test_dev_best';save(run_dir/'run.json',state)
            options=SimpleNamespace(data_dir=Path(tree['data_dir']),document_data_dir=Path(tree['document_data_dir']),
                                    model=m['model_path'],batch_size=4,seed=m['seed'])
            test,pred,merged=evaluate_checkpoint(Path(state['best_checkpoint']),'test',options,run_dir,319)
            state.update(test_prediction=str(pred),test_merged=str(merged),test_official=test['official'],
                         test_scores=test_diagnostics(tree,pred,merged))
            require(state['test_official']==state['test_scores']['official_offset'],'Test official modes differ')
            save(run_dir/'run.json',state);save(run_dir/'final-verification.json',verify(run_dir,True))
        verify_locks({**m['frozen_hashes'],**hashes})
        state.update(status='completed',current_activity='completed',wall_seconds=time.monotonic()-started,
                     finished_at=datetime.now(timezone.utc).isoformat())
        save(run_dir/'run.json',state);report(run_dir,state)
    except BaseException as error:
        state.update(status='failed',error=repr(error),finished_at=datetime.now(timezone.utc).isoformat())
        save(run_dir/'run.json',state);raise


if __name__=='__main__':main()
