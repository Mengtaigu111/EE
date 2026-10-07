"""User-requested supplementary test comparison; original dev selection unchanged."""
from datetime import datetime,timezone
import json,os,sys
from pathlib import Path
from types import SimpleNamespace

ROOT=Path('/home/myluo/EE/EXCEEDS')
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'tasks/stage2-event-tree/curriculum'))
from online_dev_config import read,save,sha,require,verify_locks
from online_dev_trainer import validate_online_checkpoint
from appendix_fixed_eval import evaluate_checkpoint
from run_sorted_first_online_dev import test_diagnostics
from verify_artifacts import _artifacts


def main():
    require('TMUX' in os.environ,'GPU evaluation must run in tmux')
    m=read(HERE/'manifest.json');verify_locks(m['locks'])
    original=Path(m['original_run']);recipe=m['recipe']
    state=dict(status='running',activity='initialize',steps=m['steps'],results={},
               original_run=str(original),original_dev_best=2205,batch_size=m['batch_size'],
               started_at=datetime.now(timezone.utc).isoformat())
    save(HERE/'state.json',state)
    args=SimpleNamespace(data_dir=Path(recipe['data_dir']),document_data_dir=Path(recipe['document_data_dir']),
                         model=recipe['model_path'],batch_size=m['batch_size'],seed=42)
    try:
        for step in m['steps']:
            state['activity']=f'test_checkpoint_{step}';save(HERE/'state.json',state)
            verify_locks(m['locks']);cp=original/'tree/checkpoints'/f'checkpoint-{step}'
            validate_online_checkpoint(cp,step,3208)
            metrics,pred,merged=evaluate_checkpoint(cp,'test',args,HERE,319)
            scores=test_diagnostics(recipe,pred,merged)
            require(scores['official_offset']==metrics['official'],'Official supplementary score differs')
            completion=read(str(pred)+'.complete.json')
            metrics_path=str(pred).removesuffix('.predictions.jsonl')+'.metrics.json'
            validation=_artifacts(recipe,original,step,pred,merged,metrics_path,
                                  metrics['official'],scores,metrics['complex'],completion,batch_size=m['batch_size'])
            state['results'][str(step)]=dict(official_offset=metrics['official'],scores=scores,
                complex=metrics['complex'],predictions=str(pred),merged=str(merged),metrics=metrics_path,
                generation=completion,verification=validation)
            verify_locks(m['locks']);save(HERE/'state.json',state)
        require(read(original/'run.json')['best_step']==2205,'Historical dev selection changed')
        state.update(status='completed',activity='completed',finished_at=datetime.now(timezone.utc).isoformat())
        save(HERE/'state.json',state)
        keys=['TI_f1','TC_f1','AI_f1','AC_f1','EC_I_f1','EC_C_f1']
        old=read(original/'run.json')
        lines=['# CoT1 supplementary test checkpoints1604/1804/3208','',
               'All rows use fixed official offset scoring, greedy batch4/max_new_tokens319. '
               'Original dev-selected checkpoint2205 and history are unchanged.','',
               '| checkpoint | TI | TC | AI | AC | EC_I | EC_C |','|---|---:|---:|---:|---:|---:|---:|']
        for step in [1604,1804,2205,3208]:
            values=old['test_official'] if step==2205 else state['results'][str(step)]['official_offset']
            lines.append('| '+str(step)+' | '+' | '.join(f'{values[k]*100:.4f}' for k in keys)+' |')
        lines+=['','All18 P/R/F1, four separate scoring modes, raw-output consistency, input/adapter hashes '
                'and7 complex metrics independently verified per checkpoint in state.json.','',
                'Batch8 benchmark on48 test examples changed2 raw/parsed trees and only saved7.35% time; '
                'batch4 was retained for consistency with existing test protocol.','']
        (HERE/'report.md').write_text('\n'.join(lines))
    except BaseException as error:
        state.update(status='failed',error=repr(error),finished_at=datetime.now(timezone.utc).isoformat())
        save(HERE/'state.json',state);raise


if __name__=='__main__':main()
