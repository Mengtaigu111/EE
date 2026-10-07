"""Fresh one-epoch shuffled Type-CoT SFT; fixed final weights, no dev controller."""
import argparse
import math
import os
from pathlib import Path
import sys
import time

sys.path.insert(0,str(Path(__file__).resolve().parent))
from config import CURRICULUM, PROJECT, read, jsonl, save, sha, require, validate_manifest, verify_locks
sys.path.insert(0,str(PROJECT/'tasks/stage2-event-tree/scripts'))
import torch
from sorted_first_sampler import SortedFirstSampler, SortedFirstTrainer, IndexedDataset, ObservedCollator


class ShuffledSampler(SortedFirstSampler):
    def __iter__(self):
        epoch=self.epoch
        indices=torch.randperm(self.size,generator=torch.Generator().manual_seed(self.seed+epoch)).tolist()
        return iter((epoch,i) for i in indices)


class CoTTrainer(SortedFirstTrainer):
    def _get_train_sampler(self,train_dataset=None):
        return ShuffledSampler(len(self.train_dataset if train_dataset is None else train_dataset),self.sampling_seed)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--smoke',action='store_true')
    args=p.parse_args()
    require('TMUX' in os.environ,'Training must run in tmux')
    os.environ.setdefault('TOKENIZERS_PARALLELISM','false')
    from peft import LoraConfig,get_peft_model
    from transformers import AutoModelForCausalLM,AutoTokenizer,TrainingArguments,TrainerCallback,set_seed
    from model_utils import AssistantCollator,audit_rows,encode_training_row,load_model_config
    from curriculum_trainer import validate_resume_checkpoint
    m=read(args.manifest);validate_manifest(m)
    require(torch.cuda.is_available() and torch.cuda.is_bf16_supported(),'CUDA bf16 required')
    set_seed(42)
    rows=jsonl(Path(m['cot_data_dir'])/'train.jsonl')
    tokenizer=AutoTokenizer.from_pretrained(m['model_path'],local_files_only=True)
    tokenizer.padding_side='right'
    lengths=[len(encode_training_row(r,tokenizer)['input_ids']) for r in rows] if args.smoke else None
    selected=sorted(sorted(range(len(rows)),key=lambda i:(-lengths[i],i))[:32]) if args.smoke else list(range(len(rows)))
    rows=[rows[i] for i in selected]
    steps,warmup=(2,1) if args.smoke else (802,25)
    token_audit=audit_rows(rows,tokenizer,4096)
    require(token_audit['overlength_count']==0,'Truncation forbidden')
    args.output.mkdir(parents=True,exist_ok=False)
    save(args.output/'token_audit.json',token_audit)
    cfg=dict(initialization='fresh base + new LoRA',init_adapter=None,model_path=m['model_path'],
             seed=42,epochs=1,max_steps=steps,warmup_steps=warmup,scheduler='cosine',learning_rate=.0001,
             r=16,alpha=32,dropout=.05,max_len=4096,precision='bf16',microbatch=1,gradient_accumulation=16,
             enable_thinking=False,loss='assistant tokens only',smoke=args.smoke,
             ordered_ids=[r['id'] for r in rows],source_indices=selected,source_token_lengths=lengths,
             train_file=str(Path(m['cot_data_dir'])/'train.jsonl'),train_sha256=sha(Path(m['cot_data_dir'])/'train.jsonl'),
             selection='fixed final checkpoint',resume_from_checkpoint=None)
    save(args.output/'config.json',cfg)
    started=time.monotonic();torch.cuda.reset_peak_memory_stats()
    model=AutoModelForCausalLM.from_pretrained(m['model_path'],local_files_only=True,
          config=load_model_config(m['model_path']),torch_dtype=torch.bfloat16,attn_implementation='sdpa')
    model.config.use_cache=False
    targets=m['tree_reference']['targets']
    model=get_peft_model(model,LoraConfig(task_type='CAUSAL_LM',r=16,lora_alpha=32,lora_dropout=.05,target_modules=targets))
    model.print_trainable_parameters()
    dataset=IndexedDataset([encode_training_row(r,tokenizer,4096) for r in rows],[r['id'] for r in rows])
    training_args=TrainingArguments(output_dir=str(args.output/'checkpoints'),
        per_device_train_batch_size=1,gradient_accumulation_steps=16,
        learning_rate=.0001,lr_scheduler_type='cosine',warmup_steps=warmup,
        adam_beta1=.9,adam_beta2=.999,adam_epsilon=1e-8,weight_decay=0.,
        num_train_epochs=1,max_steps=steps,bf16=True,gradient_checkpointing=True,
        gradient_checkpointing_kwargs={'use_reentrant':False},seed=42,data_seed=42,
        logging_steps=1,logging_first_step=True,save_strategy='epoch',save_total_limit=None,
        eval_strategy='no',report_to='none',optim='adamw_torch',remove_unused_columns=False,
        label_names=['labels'],dataloader_num_workers=0,dataloader_drop_last=False)
    collator=ObservedCollator(AssistantCollator(tokenizer.pad_token_id))
    trainer=CoTTrainer(model=model,args=training_args,train_dataset=dataset,data_collator=collator,
                      processing_class=tokenizer,observation_path=args.output/'observed-order.jsonl',sampling_seed=42)
    class InitializationCallback(TrainerCallback):
        def on_train_begin(self,args,state,control,**kwargs):
            scheduler=getattr(trainer.lr_scheduler,'scheduler',trainer.lr_scheduler)
            require(not trainer.optimizer.state_dict()['state'] and state.global_step==0,
                    'Initial CoT optimizer/trainer not empty')
            require(scheduler.last_epoch==0,'Initial CoT scheduler not at zero')
            save(args_output/'initialization.json',dict(fresh_optimizer=True,optimizer_state_entries=0,
                 scheduler_last_epoch=scheduler.last_epoch,global_step=state.global_step,
                 fresh_lora=True,adapter_source=None))
            return control
    args_output=args.output
    trainer.add_callback(InitializationCallback())
    result=trainer.train()
    torch.cuda.synchronize();elapsed=time.monotonic()-started
    require(trainer.state.global_step==steps,'CoT ended before fixed budget')
    records=jsonl(args.output/'observed-order.jsonl')
    expected=list(ShuffledSampler(len(rows),42))
    require([(r['epoch'],r['index']) for r in records]==expected,'CoT shuffled pass differs')
    require([r['microbatch'] for r in records]==list(range(len(rows))),'CoT actual batches repeat/skip')
    require(all(r['id']==dataset.ids[r['index']] and r['optimizer_step_before']==j//16
                for j,r in enumerate(records)),'CoT ID/update boundary differs')
    validate_resume_checkpoint(args.output/'checkpoints'/f'checkpoint-{steps}',steps,steps)
    trainer.save_model(str(args.output/'final-adapter'));tokenizer.save_pretrained(args.output/'final-adapter')
    trainer.state.save_to_json(str(args.output/'trainer_state.json'))
    for name in ('adapter_model.safetensors','adapter_config.json'):
        require(sha(args.output/'final-adapter'/name)==sha(args.output/'checkpoints'/f'checkpoint-{steps}'/name),
                'CoT final export differs')
    losses=[x['loss'] for x in trainer.state.log_history if 'loss' in x]
    require(losses and all(math.isfinite(x) for x in losses) and math.isfinite(result.metrics['train_loss']),
            'Nonfinite CoT loss')
    verify_locks(m['frozen_hashes'])
    metrics={**result.metrics,'status':'completed','global_step':steps,'epoch':trainer.state.epoch,
             'smoke':args.smoke,'wall_seconds':elapsed,'seconds_per_update':elapsed/steps,
             'peak_memory_bytes':torch.cuda.max_memory_allocated(),'observed_rows':len(records),
             'finite_loss':True,'all_train_rows_once':True,
             'final_adapter_sha256':sha(args.output/'final-adapter/adapter_model.safetensors')}
    save(args.output/'train_metrics.json',metrics)
    print(__import__('json').dumps(metrics),flush=True)


if __name__=='__main__':main()
