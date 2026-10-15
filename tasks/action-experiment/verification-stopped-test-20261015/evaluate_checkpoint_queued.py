"""Supplemental test of an explicit checkpoint; GPU starts as soon as memory is free."""
import argparse
import os
import subprocess
import time
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from action_common import read_json,read_rows,sha256,verify_locks,verify_data,write_json,load_module,ROOT


def select_stopped_checkpoint(run,step):
    source=read_json(run/'provenance.json')
    configuration=read_json(run/'configuration.json')
    if source['stage']!='action_grpo' or source.get('smoke') or configuration.get('smoke'):
        raise ValueError('Only a formal Action GRPO run may be selected')
    history=read_rows(run/'dev-history.jsonl')
    steps=[row['step'] for row in history]
    if not history or steps!=sorted(set(steps)):
        raise ValueError('Expected unique chronological completed dev checkpoints')
    matches=[d for d in history if d['step']==step]
    if len(matches)!=1:raise ValueError('Requested step has no completed dev evaluation')
    best=matches[0]
    checkpoint=Path(best['adapter']).resolve()
    expected=(run/'checkpoints'/f"step-{best['step']}"/'policy').resolve()
    if checkpoint!=expected:raise ValueError('Checkpoint is outside its recorded run step')
    metrics=read_json(run/'dev'/f"step-{best['step']}"/'metrics.json')
    if metrics['metrics']!=best['metrics']:
        raise ValueError('Completed dev metrics differ from selection history')
    adapter=checkpoint
    locks={str((run/name).resolve()):sha256(run/name) for name in
           ('provenance.json','configuration.json','dev-history.jsonl')}
    locks[str((run/'dev'/f'step-{step}'/'metrics.json').resolve())]=sha256(run/'dev'/f'step-{step}'/'metrics.json')
    for name in ('adapter_model.safetensors','adapter_config.json'):
        if not (adapter/name).is_file():raise FileNotFoundError(adapter/name)
        for path in (adapter/name,checkpoint/name):locks[str(path.resolve())]=sha256(path)
    return source,best,adapter,locks


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--step',type=int,required=True)
    p.add_argument('--after-exit-file',type=Path)
    args=p.parse_args();args.run=args.run.resolve();args.output=args.output.resolve()
    if 'TMUX' not in os.environ:raise RuntimeError('Run GPU evaluation in tmux')
    if args.output.exists():raise FileExistsError(args.output)
    source,best,adapter,selection_locks=select_stopped_checkpoint(args.run,args.step)
    selection_locks[str(Path(__file__).resolve())]=sha256(__file__)
    verify_locks(source['input_sha256']);verify_locks(selection_locks);verify_data(source['data_dir'])
    selection=dict(run=str(args.run),adapter=str(adapter),source_checkpoint_step=best['step'],checkpoint_requested_by_user=True,
        source_run_status='interrupted_probability_guard',source_run_completed=False,
        selection='Explicit user-requested supplementary checkpoint; dev-selected best is unchanged',
        split='test',stage='interim_test',test_controls_training=False,
        adapter_sha256=sha256(adapter/'adapter_model.safetensors'),input_sha256=selection_locks)
    write_json(args.output.with_name(args.output.name+'-inputs.json'),selection)
    import torch
    from transformers import AutoModelForCausalLM,AutoTokenizer
    from peft import PeftModel
    from action_evaluation import evaluate_model
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():raise RuntimeError('CUDA bf16 required')
    utils=load_module('_interim_test_model_utils',ROOT/'tasks/stage2-event-tree/scripts/model_utils.py')
    base=AutoModelForCausalLM.from_pretrained(source['model_path'],local_files_only=True,
        config=utils.load_model_config(source['model_path']),torch_dtype=torch.bfloat16,attn_implementation='sdpa')
    model=PeftModel.from_pretrained(base,str(adapter),is_trainable=False,local_files_only=True)
    tokenizer=AutoTokenizer.from_pretrained(source['model_path'],local_files_only=True)
    # Hash checks and CPU loading are complete before resource polling.
    # There is no requirement for a continuous empty interval.
    print('CPU_READY: wait predecessor exit if set; then check GPU every 1 second and start immediately',flush=True)
    while True:
        if args.after_exit_file is not None and not args.after_exit_file.exists():
            time.sleep(1)
            continue
        probe=subprocess.run(['nvidia-smi','--id=0','--query-gpu=memory.free','--format=csv,noheader,nounits'],capture_output=True,text=True)
        if probe.returncode==0 and probe.stdout.strip().isdigit() and int(probe.stdout.strip())>=21000:
            print(f'GPU_FREE: {probe.stdout.strip()} MiB; loading immediately',flush=True)
            try:
                model.to('cuda')
                break
            except torch.OutOfMemoryError:
                # Another job can race this observation. Preserve the model
                # and retry availability without a stability delay.
                model.to('cpu');torch.cuda.empty_cache()
                print('GPU_LOAD_RACE: memory changed; retry polling',flush=True)
        time.sleep(1)
    print('GPU_LOADED: starting full test',flush=True)
    result=evaluate_model(model,tokenizer,read_rows(Path(source['data_dir'])/'test.jsonl'),args.output,
                          max_new_tokens=768,batch_size=4)
    verify_locks(source['input_sha256']);verify_locks(selection_locks)
    write_json(args.output/'selection-source.json',selection)
    print(result['metrics'],flush=True)


if __name__=='__main__':main()
