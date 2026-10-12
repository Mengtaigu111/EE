"""Read-only input/model/tokenizer checks; no model loading or optimizer updates."""
import argparse
from pathlib import Path
import tempfile
from action_common import (DATA_DIR,MODEL_PATH,COT_ADAPTER,MAX_LENGTH,MAX_NEW_TOKENS,
                           prepare_run,read_rows,write_json,load_module,ROOT,verify_locks)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage',choices=('sft','grpo'),default='sft')
    parser.add_argument('--init-adapter',type=Path)
    parser.add_argument('--output',type=Path,required=True,help='New JSON evidence file')
    args=parser.parse_args()
    if args.output.exists():raise FileExistsError(args.output)
    adapter=args.init_adapter or COT_ADAPTER
    if args.stage=='sft':
        from train_sft import inspect_cot_source
        handoff=inspect_cot_source(adapter,MODEL_PATH)
    else:
        if args.init_adapter is None:parser.error('--init-adapter is required for GRPO')
        from train_grpo import check_sft_source
        from action_common import read_json,sha256
        handoff={'sft_metrics':str(check_sft_source(adapter,read_json,sha256,False))}
    configuration=dict(data_dir=str(DATA_DIR),max_len=MAX_LENGTH,max_new_tokens=MAX_NEW_TOKENS)
    with tempfile.TemporaryDirectory(prefix='action-preflight-') as directory:
        provenance=prepare_run(Path(directory)/'dry-run','action_'+args.stage,adapter,configuration)
    from transformers import AutoTokenizer
    tokenizer=AutoTokenizer.from_pretrained(MODEL_PATH,local_files_only=True)
    utils=load_module('_action_preflight_model_utils',ROOT/'tasks/stage2-event-tree/scripts/model_utils.py')
    rows=read_rows(DATA_DIR/'train.jsonl')
    sample=rows[:8]
    encoded=[utils.encode_training_row(r,tokenizer,MAX_LENGTH) for r in sample]
    for row in rows:
        prompt=utils.prompt_tokens(row['prompt'],tokenizer)
        if len(prompt)+MAX_NEW_TOKENS>MAX_LENGTH:
            raise ValueError(f"Prompt plus generation cap exceeds context: {row['id']}")
    verify_locks(provenance['input_sha256'])
    write_json(args.output,dict(status='passed',stage=args.stage,model_loaded=False,
               optimizer_updates=0,handoff=handoff,provenance=provenance,
               max_new_tokens=MAX_NEW_TOKENS,max_length=MAX_LENGTH,
               assistant_mask_verified_samples=len(encoded),all_train_prompt_budget_verified=True))
    print('Preflight passed; no model weights loaded and no training started.')

if __name__=='__main__':main()
