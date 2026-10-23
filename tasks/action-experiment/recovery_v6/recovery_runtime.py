"""Resume bookkeeping kept separate from unchanged GRPO mathematics."""
import copy
import json
from pathlib import Path
import re
import shutil
import time


def validate_smoke(output):
    """Validate persisted parameter digests without retaining initial snapshots."""
    metrics=json.loads((Path(output)/'train_metrics.json').read_text())
    initial=metrics.get('initial_policy_parameter_sha256')
    final=metrics.get('final_policy_parameter_sha256')
    if not all(isinstance(value,str) and re.fullmatch(r'[0-9a-fA-F]{64}',value)
               for value in (initial,final)):
        raise ValueError('Smoke requires initial and final policy parameter SHA-256 digests')
    if metrics.get('status')!='completed' or initial.lower()==final.lower() or metrics.get('reference_unchanged') is not True:
        raise ValueError('GPU CLI smoke must complete, change actor and preserve reference')


def archive_uninitialized_run(output):
    """Preserve a failed initialization only when no recovery point was published."""
    output=Path(output)
    points=output/'recovery-checkpoints'
    latest=points/'latest.json'
    published=any(path.parent.is_dir() and not path.parent.name.startswith('.')
                  for path in points.glob('*/manifest.json'))
    if latest.exists() or latest.is_symlink() or published:
        raise RuntimeError('Published recovery checkpoint requires explicit resume: '+str(output))
    marker=output/'initialization.json'
    initializing=False
    if marker.is_file():
        try:
            value=json.loads(marker.read_text())
            initializing=isinstance(value,dict) and value.get('format')=='grpo-initialization-v1' and value.get('status')=='initializing'
        except (OSError,ValueError):
            pass
    if any(output.glob('*.jsonl')) and not initializing:
        raise RuntimeError('Existing training artifacts without a recovery point: '+str(output))
    archived=output.with_name(output.name+'-pretraining-interrupted-'+str(time.time_ns()))
    output.rename(archived)
    return archived


def restore_objects(obj, state):
    obj.__dict__.update(copy.deepcopy({k:v for k,v in state.items() if k!='stop'}))


def log_offsets(output):
    return {p.name:p.stat().st_size for p in Path(output).glob('*.jsonl')}


def truncate_logs(output, offsets):
    output=Path(output)
    for name,size in offsets.items():
        if Path(name).name!=name or (size and not (output/name).is_file()):raise ValueError('Missing recovery log: '+name)
    for path in output.glob('*.jsonl'):
        size=offsets.get(path.name,0)
        if size>path.stat().st_size:raise ValueError('Recovery log shorter than checkpoint: '+path.name)
    for path in output.glob('*.jsonl'):
        size=offsets.get(path.name,0)
        if path.stat().st_size>size:
            archive=output/'interrupted-logs'/str(time.time_ns())
            archive.mkdir(parents=True)
            with path.open('rb') as source, (archive/path.name).open('wb') as target:
                source.seek(size);shutil.copyfileobj(source,target)
            with path.open('r+b') as stream:stream.truncate(size)


def import_policy(model, path):
    from peft import set_peft_model_state_dict
    from peft.utils.save_and_load import load_peft_weights
    from grpo_core import activate_adapter
    weights=load_peft_weights(str(path),device='cpu')
    result=set_peft_model_state_dict(model,weights,adapter_name='policy')
    if result.unexpected_keys:raise ValueError('Unexpected adapter weights: '+str(result.unexpected_keys))
    if any('.policy.' in name for name in result.missing_keys):raise ValueError('Missing policy adapter weights')
    activate_adapter(model,'policy')


def validate_warm_start(run, step, sft_adapter, configuration):
    from action_common import read_json, read_rows, verify_locks
    run=Path(run).resolve()
    provenance=read_json(run/'provenance.json');old=read_json(run/'configuration.json')
    verify_locks(provenance['input_sha256'])
    if Path(provenance['init_adapter']).resolve()!=Path(sft_adapter).resolve():raise ValueError('SFT reference changed')
    if old.get('smoke'):raise ValueError('Formal warm-start cannot use smoke')
    for key in ('reward_version','policy_update_mask_mode','gradient_accumulation','dev_steps','max_optimizer_steps','early_stopping_patience'):
        if key=='max_optimizer_steps' and configuration.get('smoke'):continue
        if old.get(key)!=configuration.get(key):raise ValueError('Warm-start configuration differs: '+key)
    history=[d for d in read_rows(run/'dev-history.jsonl') if d['step']<=step]
    if not history or history[-1]['step']!=step:raise ValueError('Warm start needs complete dev checkpoint')
    if history[-1]['metrics']!=read_json(run/'dev'/f'step-{step}'/'metrics.json')['metrics']:raise ValueError('Dev history mismatch')
    adapter=run/'checkpoints'/f'step-{step}'/'policy'
    if Path(history[-1]['adapter']).resolve()!=adapter.resolve():raise ValueError('Adapter path differs')
    files=[run/'provenance.json',run/'configuration.json',run/'dev-history.jsonl',run/'groups.jsonl',
           run/'planned-train-order.json',run/'dev'/f'step-{step}'/'metrics.json']
    files+=list(adapter.glob('*'))
    if not (adapter/'adapter_model.safetensors').is_file():raise FileNotFoundError(adapter)
    return dict(adapter=adapter,history=history,locked_files=[p for p in files if p.is_file()])


def inherit_prefix(source, output, warm, tokenizer, model):
    from action_common import read_rows,write_json,append_jsonl
    from grpo_core import save_policy
    source=Path(source);output=Path(output)
    step=warm['history'][-1]['step']
    config=json.loads((source/'configuration.json').read_text())
    count=step*config['gradient_accumulation']
    expected=json.loads((source/'planned-train-order.json').read_text())['ids'][:count]
    actual=json.loads((output/'planned-train-order.json').read_text())['ids'][:count]
    if actual!=expected:raise ValueError('Warm-start train order differs')
    rows=[]
    with (source/'groups.jsonl').open() as stream:
        for _ in range(count):
            line=next(stream,None)
            if line is None:raise ValueError('Missing inherited train prefix')
            rows.append(json.loads(line))
    if [r['id'] for r in rows]!=expected or [r['group_index'] for r in rows]!=list(range(count)):
        raise ValueError('Inherited observed train prefix differs')
    with (output/'groups.jsonl').open('w') as stream:
        for row in rows:stream.write(json.dumps(row)+'\n')
    checkpoint=save_policy(model,tokenizer,output/'checkpoints'/f'step-{step}')
    history=warm['history']
    for record in history:
        record=copy.deepcopy(record)
        shutil.copytree(source/'dev'/f"step-{record['step']}",output/'dev'/f"step-{record['step']}")
        if record['step']==step:record['adapter']=str(checkpoint)
        record['inherited_from']=str(source.resolve())
        append_jsonl(output/'dev-history.jsonl',record)
    best=max(history,key=lambda d:(d['metrics']['TC_f1'],d['metrics']['TI_f1'],-d['step']))
    shutil.copytree(Path(best['adapter']),output/'best-adapter')
    return dict(best_step=best['step'],best_key=[best['metrics']['TC_f1'],best['metrics']['TI_f1']],
        early_stopping=history[-1]['early_stopping'],active_groups=sum(r['active'] for r in rows),
        sampled_tokens=rows[-1]['sampled_tokens'])


def reconcile_artifacts(output, last_dev_step, best_step):
    """Archive partial/uncommitted dev products before re-running evaluation."""
    output=Path(output)
    for parent in ('dev','checkpoints'):
        for path in (output/parent).glob('step-*'):
            if int(path.name[5:])>last_dev_step:
                dest=output/'interrupted-artifacts'/str(time.time_ns())/parent/path.name
                dest.parent.mkdir(parents=True,exist_ok=True);shutil.move(str(path),dest)
    if best_step is None:return
    best=output/'best-adapter'
    source=output/'checkpoints'/f'step-{best_step}'/'policy'
    if not source.is_dir():
        # Inherited early dev entries can refer to their frozen original run.
        history=[json.loads(line) for line in (output/'dev-history.jsonl').open()]
        source=Path(next(d['adapter'] for d in history if d['step']==best_step))
    if best.exists():shutil.rmtree(best)
    shutil.copytree(source,best)


def prune_checkpoints(root, keep=3):
    from checkpoint_state import resolve_latest
    root=Path(root);latest=resolve_latest(root)
    candidates=sorted((p for p in root.iterdir() if p.is_dir() and not p.name.startswith('.')),
                      key=lambda p:p.stat().st_mtime_ns,reverse=True)
    for path in candidates[keep:]:
        if path!=latest:shutil.rmtree(path)


def resume_decision(step, cursor, total, last_dev_step, stopped, dev_steps):
    if stopped:return dict(evaluate=False,train=False)
    due=last_dev_step<step and (step==0 or step%dev_steps==0 or cursor==total)
    return dict(evaluate=due,train=cursor<total)
