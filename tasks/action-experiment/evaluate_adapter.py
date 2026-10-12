"""Evaluate only the dev-selected adapter from a completed Action SFT/GRPO run."""
import argparse
import os
from pathlib import Path
from action_common import (MAX_NEW_TOKENS,read_json,sha256,verify_locks,verify_data,
                           read_rows,load_module,ROOT,write_json)


def selected_adapter(run,allow_smoke=False):
    run=Path(run)
    metrics=read_json(run/'train_metrics.json')
    if metrics.get('status')!='completed' or metrics.get('stage') not in ('action_sft','action_grpo'):
        raise ValueError('Evaluation requires a completed Action run')
    if metrics.get('smoke') and not allow_smoke:
        raise ValueError('Cannot report a smoke checkpoint as formal evaluation')
    adapter=run/'best-adapter'
    if metrics.get('best_adapter_sha256')!=sha256(adapter/'adapter_model.safetensors'):
        raise ValueError('Selected adapter content changed')
    if metrics.get('best_adapter_config_sha256')!=sha256(adapter/'adapter_config.json'):
        raise ValueError('Selected adapter config changed or unfrozen')
    return adapter


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--split',choices=('dev','test'),default='test')
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--max-new-tokens',type=int,default=MAX_NEW_TOKENS)
    parser.add_argument('--batch-size',type=int,default=4)
    parser.add_argument('--allow-smoke',action='store_true')
    args=parser.parse_args()
    if args.output.exists():raise FileExistsError(args.output)
    if 'TMUX' not in os.environ:raise ValueError('GPU evaluation must run inside tmux')
    adapter=selected_adapter(args.run,args.allow_smoke)
    source=read_json(args.run/'provenance.json')
    verify_locks(source['input_sha256']);verify_data(source['data_dir'])
    import torch
    from transformers import AutoModelForCausalLM,AutoTokenizer
    from peft import PeftModel
    from action_evaluation import evaluate_model
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():raise ValueError('CUDA bf16 required')
    utils=load_module('_action_eval_model_utils',ROOT/'tasks/stage2-event-tree/scripts/model_utils.py')
    base=AutoModelForCausalLM.from_pretrained(source['model_path'],local_files_only=True,
        config=utils.load_model_config(source['model_path']),torch_dtype=torch.bfloat16,attn_implementation='sdpa').to('cuda')
    model=PeftModel.from_pretrained(base,str(adapter),is_trainable=False,local_files_only=True)
    tokenizer=AutoTokenizer.from_pretrained(source['model_path'],local_files_only=True)
    rows=read_rows(Path(source['data_dir'])/f'{args.split}.jsonl')
    result=evaluate_model(model,tokenizer,rows,args.output,max_new_tokens=args.max_new_tokens,batch_size=args.batch_size)
    verify_locks(source['input_sha256'])
    write_json(args.output/'selection-source.json',dict(run=str(args.run.resolve()),adapter=str(adapter.resolve()),
        adapter_sha256=sha256(adapter/'adapter_model.safetensors'),split=args.split,
        source_best_step=read_json(args.run/'train_metrics.json')['best_step'],
        smoke=read_json(args.run/'train_metrics.json').get('smoke',False),test_controls_training=False))
    print(result['metrics'])

if __name__=='__main__':main()
