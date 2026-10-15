"""Read-only test of the best completed dev checkpoint of an interrupted run."""
import argparse
import os
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from action_common import read_json,read_rows,sha256,verify_locks,verify_data,write_json,load_module,ROOT


def select_stopped_checkpoint(run):
    source=read_json(run/'provenance.json')
    configuration=read_json(run/'configuration.json')
    if source['stage']!='action_grpo' or source.get('smoke') or configuration.get('smoke'):
        raise ValueError('Only a formal Action GRPO run may be selected')
    history=read_rows(run/'dev-history.jsonl')
    steps=[row['step'] for row in history]
    if not history or steps!=sorted(set(steps)):
        raise ValueError('Expected unique chronological completed dev checkpoints')
    best=max(history,key=lambda d:(d['metrics']['TC_f1'],d['metrics']['TI_f1']))
    checkpoint=Path(best['adapter']).resolve()
    expected=(run/'checkpoints'/f"step-{best['step']}"/'policy').resolve()
    if checkpoint!=expected:raise ValueError('Checkpoint is outside its recorded run step')
    metrics=read_json(run/'dev'/f"step-{best['step']}"/'metrics.json')
    if metrics['metrics']!=best['metrics']:
        raise ValueError('Completed dev metrics differ from selection history')
    adapter=run/'best-adapter'
    locks={str((run/name).resolve()):sha256(run/name) for name in
           ('provenance.json','configuration.json','dev-history.jsonl')}
    for name in ('adapter_model.safetensors','adapter_config.json'):
        if sha256(adapter/name)!=sha256(checkpoint/name):
            raise ValueError('Best adapter differs from the dev-selected checkpoint')
        for path in (adapter/name,checkpoint/name):locks[str(path.resolve())]=sha256(path)
    return source,best,adapter,locks


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();args.run=args.run.resolve();args.output=args.output.resolve()
    if 'TMUX' not in os.environ:raise RuntimeError('Run GPU evaluation in tmux')
    if args.output.exists():raise FileExistsError(args.output)
    source,best,adapter,selection_locks=select_stopped_checkpoint(args.run)
    selection_locks[str(Path(__file__).resolve())]=sha256(__file__)
    verify_locks(source['input_sha256']);verify_locks(selection_locks);verify_data(source['data_dir'])
    selection=dict(run=str(args.run),adapter=str(adapter),source_best_step=best['step'],
        source_run_status='interrupted_probability_guard',source_run_completed=False,
        selection='TC then TI over completed dev checkpoints; earlier on exact ties',
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
        config=utils.load_model_config(source['model_path']),torch_dtype=torch.bfloat16,attn_implementation='sdpa').to('cuda')
    model=PeftModel.from_pretrained(base,str(adapter),is_trainable=False,local_files_only=True)
    tokenizer=AutoTokenizer.from_pretrained(source['model_path'],local_files_only=True)
    result=evaluate_model(model,tokenizer,read_rows(Path(source['data_dir'])/'test.jsonl'),args.output,
                          max_new_tokens=768,batch_size=4)
    verify_locks(source['input_sha256']);verify_locks(selection_locks)
    write_json(args.output/'selection-source.json',selection)
    print(result['metrics'],flush=True)


if __name__=='__main__':main()
