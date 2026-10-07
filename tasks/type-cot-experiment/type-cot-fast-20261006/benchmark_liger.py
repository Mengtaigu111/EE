"""Isolated fresh-model GPU training trials; never updates historical adapters."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import random
import subprocess
import sys
import time

PROJECT=Path(__file__).resolve().parents[3]
WARMUP_UPDATES=2
MEASURED_UPDATES=8


def save(path,value):
    path=Path(path)
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def select_trial_indices(lengths):
    if len(lengths)<160:raise ValueError('Need at least 160 train rows')
    # Warmup 32 representative rows; then two updates stress the longest32.
    longest=sorted(range(len(lengths)),key=lambda i:(-lengths[i],i))[:32]
    representative=random.Random(42).sample(range(len(lengths)),128)
    return representative[:32]+longest+representative[32:]


def update_groups(indices):
    if len(indices)%16:raise ValueError('Incomplete effective batch')
    return [indices[i:i+16] for i in range(0,len(indices),16)]


def assistant_token_loss(logits,labels,num_items_in_batch):
    """CPU reference: native causal CE summed over all effective-batch targets."""
    import torch.nn.functional as F
    return F.cross_entropy(logits[:,:-1,:].reshape(-1,logits.shape[-1]),
                           labels[:,1:].reshape(-1),ignore_index=-100,reduction='sum')/num_items_in_batch


def choose_microbatch(baseline,candidate):
    if baseline.get('status')!='completed':raise RuntimeError('Baseline training failed; cannot launch')
    stable=(candidate.get('status')=='completed' and candidate.get('finite_loss') is True
            and candidate.get('finite_gradients') is True and candidate.get('longest_stress_passed') is True
            and candidate.get('memory_free_at_peak',0)>=256*1024**2)
    savings=1-candidate['seconds_per_update']/baseline['seconds_per_update'] if candidate.get('status')=='completed' else None
    selected=2 if stable and savings>=.05 else 1
    return dict(microbatch=selected,gradient_accumulation=16//selected,
                time_savings_fraction=savings,stable_candidate=stable,
                reason='stable and >=5% faster' if selected==2 else 'retain baseline: speed/stability/memory gate not met')


def run_trial(manifest,output,phase,microbatch):
    import torch
    from torch.utils.data import Dataset,SequentialSampler
    from peft import LoraConfig,get_peft_model
    from transformers import AutoModelForCausalLM,AutoTokenizer,Trainer,TrainingArguments,TrainerCallback,set_seed
    sys.path.insert(0,str(PROJECT/'tasks/stage2-event-tree/scripts'))
    from model_utils import AssistantCollator,encode_training_row,load_model_config
    os.environ.setdefault('TOKENIZERS_PARALLELISM','false')
    output.mkdir(parents=True,exist_ok=False)
    m=json.loads(manifest.read_text());ref=m['tree_reference']
    source=Path(m['cot_data_dir'])/'train.jsonl' if phase=='cot' else Path(ref['train_file'])
    rows=[json.loads(line) for line in source.read_text().splitlines()]
    if not all(row.get('split','train')=='train' for row in rows):raise ValueError('Only train allowed')
    set_seed(42)
    tokenizer=AutoTokenizer.from_pretrained(m['model_path'],local_files_only=True)
    tokenizer.padding_side='right'
    encoded=[encode_training_row(row,tokenizer,4096) for row in rows]
    indices=select_trial_indices([len(row['input_ids']) for row in encoded])
    groups=update_groups(indices)
    save(output/'config.json',dict(phase=phase,microbatch=microbatch,gradient_accumulation=16//microbatch,
        warmup_updates=WARMUP_UPDATES,measured_updates=MEASURED_UPDATES,seed=42,
        training_kernel='liger',initialization='fresh base + new LoRA',adapter_source=None,model_path=m['model_path'],
        precision='bf16',attention='sdpa',gradient_checkpointing=True,train_file=str(source),
        train_sha256=sha(source),manifest_sha256=sha(manifest),benchmark_sha256=sha(__file__),
        update_groups=[[rows[i]['id'] for i in group] for group in groups],
        lengths=[[len(encoded[i]['input_ids']) for i in group] for group in groups]))
    class TrialDataset(Dataset):
        def __len__(self):return len(indices)
        def __getitem__(self,index):return encoded[indices[index]]
    class TrialTrainer(Trainer):
        def _get_train_sampler(self,train_dataset=None):return SequentialSampler(self.train_dataset)
        def get_batch_samples(self,epoch_iterator,num_batches,device):
            batches,count=super().get_batch_samples(epoch_iterator,num_batches,device)
            if count is None:raise RuntimeError('Native loss kwargs token counting inactive')
            if any(torch.any(batch['labels'][:,0]!=-100) for batch in batches):
                raise RuntimeError('Shifted causal token count would differ')
            return batches,count
    from kernel_support import apply_kernel
    kernel_metadata=apply_kernel('liger')
    model=AutoModelForCausalLM.from_pretrained(m['model_path'],local_files_only=True,
            config=load_model_config(m['model_path']),torch_dtype=torch.bfloat16,attn_implementation='sdpa')
    model.config.use_cache=False
    model=get_peft_model(model,LoraConfig(task_type='CAUSAL_LM',r=16,lora_alpha=32,lora_dropout=.05,target_modules=ref['targets']))
    args=TrainingArguments(output_dir=str(output/'scratch'),per_device_train_batch_size=microbatch,
        gradient_accumulation_steps=16//microbatch,max_steps=WARMUP_UPDATES+MEASURED_UPDATES,
        learning_rate=1e-4,lr_scheduler_type='constant',warmup_steps=0,bf16=True,
        gradient_checkpointing=True,gradient_checkpointing_kwargs={'use_reentrant':False},
        seed=42,data_seed=42,logging_steps=1,logging_nan_inf_filter=False,
        save_strategy='no',eval_strategy='no',report_to='none',optim='adamw_torch',
        adam_beta1=.9,adam_beta2=.999,adam_epsilon=1e-8,weight_decay=0.,
        dataloader_num_workers=0,dataloader_drop_last=False,remove_unused_columns=False,label_names=['labels'])
    trainer=TrialTrainer(model=model,args=args,train_dataset=TrialDataset(),
                         data_collator=AssistantCollator(tokenizer.pad_token_id),processing_class=tokenizer)
    if not trainer.model_accepts_loss_kwargs:raise RuntimeError('Loss normalization must use effective-batch token count')
    timings=[];gradient_checks=[];start=[None];external=[0]
    class TimingCallback(TrainerCallback):
        def on_step_begin(self,args,state,control,**kwargs):
            torch.cuda.synchronize()
            if state.global_step==WARMUP_UPDATES:
                torch.cuda.reset_peak_memory_stats()
                free,total=torch.cuda.mem_get_info()
                external[0]=max(0,total-free-torch.cuda.memory_reserved())
            start[0]=time.monotonic()
        def on_pre_optimizer_step(self,args,state,control,**kwargs):
            finite=all(bool(torch.isfinite(p.grad).all()) for p in model.parameters() if p.requires_grad and p.grad is not None)
            gradient_checks.append(finite)
            if not finite:raise FloatingPointError('Nonfinite gradient')
        def on_step_end(self,args,state,control,**kwargs):
            torch.cuda.synchronize()
            if state.global_step>WARMUP_UPDATES:timings.append(time.monotonic()-start[0])
    trainer.add_callback(TimingCallback())
    trainer.train()
    losses=[entry['loss'] for entry in trainer.state.log_history if 'loss' in entry]
    if len(timings)!=MEASURED_UPDATES:raise RuntimeError('Incomplete measured updates')
    total=torch.cuda.get_device_properties(0).total_memory
    result=dict(status='completed',phase=phase,microbatch=microbatch,kernel_metadata=kernel_metadata,
        optimizer_updates=trainer.state.global_step,warmup_updates=WARMUP_UPDATES,
        measured_updates=len(timings),update_seconds=timings,seconds_per_update=sum(timings)/len(timings),
        representative_seconds_per_update=sum(timings[2:])/len(timings[2:]),
        longest_stress_seconds=timings[:2],longest_stress_passed=True,
        finite_loss=bool(losses) and all(math.isfinite(loss) for loss in losses),
        finite_gradients=len(gradient_checks)==trainer.state.global_step and all(gradient_checks),losses=losses,
        peak_allocated=torch.cuda.max_memory_allocated(),peak_reserved=torch.cuda.max_memory_reserved(),
        memory_free_at_peak=total-external[0]-torch.cuda.max_memory_reserved(),
        token_normalization='native num_items_in_batch sums assistant tokens across all accumulated microbatches')
    if not result['finite_loss'] or not result['finite_gradients']:raise FloatingPointError('Nonfinite trial')
    save(output/'result.json',result)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--phase',choices=('cot','tree'));parser.add_argument('--microbatch',type=int,choices=(1,2))
    args=parser.parse_args()
    if 'TMUX' not in os.environ:raise RuntimeError('GPU benchmark must run in tmux')
    if args.phase:
        try:run_trial(args.manifest,args.output,args.phase,args.microbatch)
        except Exception as exc:
            if args.output.exists():save(args.output/'result.json',dict(status='failed',exception_type=type(exc).__name__,error=str(exc),phase=args.phase,microbatch=args.microbatch))
            raise
        return
    args.output.mkdir(parents=True,exist_ok=False)
    decisions={}
    for phase in ('cot','tree'):
        results=[]
        for microbatch in (1,2):
            trial=args.output/f'{phase}-batch{microbatch}'
            with (args.output/f'{phase}-batch{microbatch}.log').open('x') as log:
                proc=subprocess.run([sys.executable,str(Path(__file__).resolve()),'--manifest',str(args.manifest.resolve()),
                    '--output',str(trial.resolve()),'--phase',phase,'--microbatch',str(microbatch)],stdout=log,stderr=subprocess.STDOUT)
            result=json.loads((trial/'result.json').read_text()) if (trial/'result.json').exists() else dict(status='failed',error=f'exit {proc.returncode}')
            results.append(result)
            if microbatch==1 and result.get('status')!='completed':raise RuntimeError(f'{phase} baseline failed')
        decisions[phase]=choose_microbatch(*results)
    save(args.output/'decision.json',dict(status='completed',phases=decisions,manifest_sha256=sha(args.manifest),benchmark_sha256=sha(__file__)))


if __name__=='__main__':main()
