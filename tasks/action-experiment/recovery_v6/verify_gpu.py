"""Real-weight arithmetic, memory, backward and complete-resume acceptance.

Writes only verification artifacts. Never publishes a formal training run.
"""
import copy
import importlib.util
import json
import os
from pathlib import Path
import random
import sys
import time

HERE=Path(__file__).resolve().parent;ACTION=HERE.parent
sys.path[:0]=[str(HERE),str(ACTION)]
OUT=ACTION/'verification-stable-v6-20261021'


def write(name,value):
    (OUT/name).write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n')


def main():
    assert 'TMUX' in os.environ
    OUT.mkdir(exist_ok=True)
    import torch
    from transformers import AutoModelForCausalLM,AutoTokenizer,GenerationConfig,set_seed
    from stable_arithmetic import initialize_dual_adapters_cpu,install_stable_arithmetic,completion_logprobs,ARITHMETIC_CONFIGURATION
    from idle_resource_loader import wait_for_idle_training_gpu
    from action_common import MODEL_PATH,ROOT,DATA_DIR,MAX_NEW_TOKENS,read_rows,generation_arguments,sha256
    from grpo_core import activate_adapter,completion_mask,behavior_logprobs,group_advantages,grpo_loss,save_policy
    from grpo_diagnostics import ProbabilityGuard
    from action_reward_v2 import reward_action_r0_v2
    from action_update_mask import build_field_mask
    from recovery_runtime import import_policy,restore_objects
    from checkpoint_state import load_checkpoint,save_checkpoint
    from train_grpo_recoverable import adapter_digest,prompt_tokens
    torch.set_num_threads(2);set_seed(42)
    started=time.monotonic()
    result=dict(status='running',arithmetic=ARITHMETIC_CONFIGURATION,stages={})
    write('result.json',result)
    tokenizer=AutoTokenizer.from_pretrained(MODEL_PATH,local_files_only=True)
    if tokenizer.pad_token_id is None:tokenizer.pad_token_id=tokenizer.eos_token_id
    train=read_rows(DATA_DIR/'train.jsonl');random.Random(42).shuffle(train)
    dev=read_rows(DATA_DIR/'dev.jsonl')
    longest={}
    for split,rows in [('train',train),('dev',dev)]:
        lengths=[prompt_tokens(tokenizer,row,'cpu').numel() for row in rows]
        index=max(range(len(rows)),key=lengths.__getitem__)
        longest[split]=dict(id=rows[index]['id'],prompt_length=lengths[index],row=rows[index])
    write('longest-inputs.json',longest)
    spec=importlib.util.spec_from_file_location('_stable_verify_config',ROOT/'tasks/stage2-event-tree/scripts/model_utils.py')
    utils=importlib.util.module_from_spec(spec);spec.loader.exec_module(utils)
    base=AutoModelForCausalLM.from_pretrained(MODEL_PATH,local_files_only=True,
        config=utils.load_model_config(MODEL_PATH),torch_dtype=torch.bfloat16,attn_implementation='sdpa')
    model=initialize_dual_adapters_cpu(base,ACTION/'runs/action-sft-v2-42-20261012/best-adapter')
    install_stable_arithmetic(model)
    assert not torch.cuda.is_initialized()
    wait_for_idle_training_gpu(lambda:model.to('cuda'),lambda:model.to('cpu'),
        on_state=lambda s:write('resource-state.json',dict(s,pid=os.getpid(),updated_unix=time.time())))
    model.config.use_cache=False
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
    model.enable_input_require_grads()
    optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=1e-5,betas=(.9,.999),eps=1e-8,weight_decay=0.)
    reference_digest=adapter_digest(model,'reference')
    runs={'A':ACTION/'runs/r0-v2-A800-recoverable-v5-42-20261019','B':ACTION/'runs/r0-v2-B-recoverable-v5-42-20261019'}
    fixtures=json.loads((ACTION/'diagnosis-AB-20261021/validation-fixtures.json').read_text())
    def last_record(path):
        with path.open('rb') as f:
            f.seek(max(0,path.stat().st_size-600000));return json.loads(f.read().splitlines()[-1])
    def point(label):
        return runs[label]/'recovery-checkpoints'/json.loads((runs[label]/'recovery-checkpoints/latest.json').read_text())['checkpoint']
    def forced_cached(prompt,ids):
        past=None;values=[]
        for i in range(ids.shape[1]):
            out=model(input_ids=prompt if i==0 else ids[:,i-1:i],attention_mask=torch.ones((1,prompt.shape[1]+i),device='cuda',dtype=torch.long),past_key_values=past,use_cache=True,logits_to_keep=1)
            logits=out.logits[:,-1];values.append(logits.gather(-1,ids[:,i:i+1]).squeeze(-1)-logits.logsumexp(-1));past=out.past_key_values;del out
        del past
        return torch.stack(values,-1)
    print('Stage 1: six real failure/control forwards using production module',flush=True)
    forward_results=[]
    with torch.no_grad():
        for label in ('A','B'):
            import_policy(model,point(label)/'adapter/policy');model.eval()
            records=[last_record(runs[label]/'probability-token-log.jsonl')]+[f['tokens'] for f in fixtures[label]]
            for row in records:
                prompt=torch.tensor([row['prompt_tokens']],device='cuda');ids=torch.tensor([row['completion_tokens']],device='cuda');mask=torch.tensor([row['completion_mask']],device='cuda',dtype=torch.bool)
                full=completion_logprobs(model,prompt,ids,mask);cached=forced_cached(prompt,ids)
                maximum=float((full-cached).abs()[mask].max());assert maximum<.001
                control=ProbabilityGuard().observe(full,cached,mask);assert not control['sample_warning']
                forward_results.append(dict(run=label,group=row['group_index'],sample=row['sample_index'],max_difference=maximum,tokens=int(mask.sum())))
    result['stages']['real_forward']=forward_results;write('result.json',result)
    # Restore the original complete B boundary; optimizer/RNG are real, not reset.
    payload=load_checkpoint(point('B'),model,optimizer,load_policy=lambda m,p:import_policy(m,p/'policy'))
    assert payload['step']==1490 and payload['next_group']==5960 and payload['pending']==0
    assert adapter_digest(model,'reference')==reference_digest==payload['reference_sha256']
    initial_policy=adapter_digest(model,'policy')
    eos=model.generation_config.eos_token_id or tokenizer.eos_token_id;eos_ids=[eos] if isinstance(eos,int) else list(eos)
    generation=GenerationConfig(do_sample=True,temperature=1.,top_p=1.,top_k=0,max_new_tokens=MAX_NEW_TOKENS,
        eos_token_id=eos,pad_token_id=tokenizer.pad_token_id,bos_token_id=tokenizer.bos_token_id,
        repetition_penalty=1.,use_cache=True,output_scores=True,return_dict_in_generate=True)
    guards={name:ProbabilityGuard() for name in ('current_vs_old','cached_vs_eval','eval_vs_train')}
    for name,guard in guards.items():restore_objects(guard,payload['probability_guards'][name])
    optimizer.zero_grad(set_to_none=True)
    print('Stage 2: real G4 x accumulation4, original reward/mask/guards and AdamW',flush=True)
    observed=[]
    for group in range(5960,5964):
        row=train[group];prompt=prompt_tokens(tokenizer,row,'cuda');samples=[]
        model.eval();activate_adapter(model,'policy')
        with torch.no_grad():
            for sample in range(4):
                generated=model.generate(input_ids=prompt,attention_mask=torch.ones_like(prompt),**generation_arguments(generation))
                ids=generated.sequences[:,prompt.shape[1]:].detach();mask=completion_mask(ids,eos_ids)
                old=behavior_logprobs(generated,prompt.shape[1],mask);del generated
                teacher=completion_logprobs(model,prompt,ids,mask)
                activate_adapter(model,'reference');ref=completion_logprobs(model,prompt,ids,mask);activate_adapter(model,'policy')
                text=tokenizer.decode(ids[0][mask[0]],skip_special_tokens=True)
                truncated=not any(int(ids[0,int(mask.sum())-1])==e for e in eos_ids)
                reward=reward_action_r0_v2(text,row,truncated=truncated,has_eos=not truncated)
                samples.append((ids,mask,old,teacher,ref,reward,text))
        advantages=group_advantages(torch.tensor([s[5]['reward'] for s in samples],device='cuda'))
        model.train();activate_adapter(model,'policy')
        for sample,(ids,mask,old,teacher,ref,reward,text) in enumerate(samples):
            current=completion_logprobs(model,prompt,ids,mask)
            guard_results={name:guards[name].observe(a,b,mask) for name,a,b in (
                ('current_vs_old',current,old),('cached_vs_eval',teacher,old),('eval_vs_train',current,teacher))}
            field_mask,_=build_field_mask(tokenizer,ids,mask,text,reward,float(advantages[sample]))
            loss,stats=grpo_loss(current,old,ref,mask,advantages[sample:sample+1],field_mask=field_mask)
            (loss/16).backward()
            observed.append(dict(group=group,sample=sample,tokens=int(mask.sum()),loss=float(loss.detach()),
                max_difference=guard_results['current_vs_old']['sample_statistics']['abs_difference_max']))
            del current,loss
        del samples
        print('Completed backward group',group,flush=True)
    norm=torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],1.)
    assert torch.isfinite(norm)
    optimizer.step();optimizer.zero_grad(set_to_none=True)
    assert adapter_digest(model,'policy')!=initial_policy
    assert adapter_digest(model,'reference')==reference_digest
    result['stages']['optimizer']=dict(samples=observed,gradient_norm=float(norm),initial_step=1490,final_step=1491,
        policy_changed=True,reference_unchanged=True)
    write('result.json',result)
    print('Stage 3: complete snapshot load and RNG equality',flush=True)
    snapshot_payload=copy.deepcopy(payload);snapshot_payload.update(step=1491,next_group=5964,pending=0,
        arithmetic=ARITHMETIC_CONFIGURATION,verification_only=True,probability_guards={k:copy.deepcopy(v.__dict__) for k,v in guards.items()})
    checkpoint=save_checkpoint(OUT/'checkpoints',f'verification-step1491-{time.time_ns()}',model,optimizer,snapshot_payload,
        export_policy=lambda m,p:save_policy(m,tokenizer,p))
    expected=torch.rand(8,device='cuda');saved_digest=adapter_digest(model,'policy')
    restored=load_checkpoint(checkpoint,model,optimizer,load_policy=lambda m,p:import_policy(m,p/'policy'))
    actual=torch.rand(8,device='cuda');assert torch.equal(expected,actual)
    assert restored['step']==1491 and restored['next_group']==5964 and adapter_digest(model,'policy')==saved_digest
    result['stages']['checkpoint']=dict(path=str(checkpoint),rng_equal=True,policy_equal=True,step=1491)
    write('result.json',result)
    print('Stage 4: longest real train prompt plus full completion budget backward',flush=True)
    optimizer.zero_grad(set_to_none=True);model.train();activate_adapter(model,'policy')
    long_prompt=prompt_tokens(tokenizer,longest['train']['row'],'cuda')
    stress_ids=torch.full((1,MAX_NEW_TOKENS),tokenizer.eos_token_id,device='cuda',dtype=torch.long)
    stress_mask=torch.ones_like(stress_ids,dtype=torch.bool)
    torch.cuda.reset_peak_memory_stats()
    score=completion_logprobs(model,long_prompt,stress_ids,stress_mask)
    (-score.mean()).backward();assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
    train_peak=torch.cuda.max_memory_allocated();del score
    optimizer.zero_grad(set_to_none=True);torch.cuda.empty_cache()
    result['stages']['longest_train']=dict(prompt_tokens=long_prompt.numel(),completion_tokens=MAX_NEW_TOKENS,
        total_tokens=long_prompt.numel()+MAX_NEW_TOKENS,peak_allocated_bytes=train_peak,finite_gradients=True,
        caveat='synthetic EOS labels only for memory stress; no optimizer update or metric')
    write('result.json',result)
    print('Stage 5: longest real dev prompt, batch1/2/4 throughput and memory',flush=True)
    model.eval();dev_prompt=prompt_tokens(tokenizer,longest['dev']['row'],'cuda')
    benchmarks=[]
    for batch_size in (1,2,4):
        torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize()
        tick=time.monotonic();benchmark=dict(batch_size=batch_size,total_rows=4,forced_new_tokens=64)
        try:
            with torch.inference_mode():
                for _ in range(4//batch_size):
                    batch=dev_prompt.repeat(batch_size,1)
                    generated=model.generate(input_ids=batch,attention_mask=torch.ones_like(batch),do_sample=False,
                        min_new_tokens=64,max_new_tokens=64,use_cache=True,pad_token_id=tokenizer.pad_token_id,
                        eos_token_id=eos,use_model_defaults=False)
                    assert generated.shape[1]-batch.shape[1]==64
                    del generated
            torch.cuda.synchronize()
            benchmark.update(status='passed',wall_seconds=time.monotonic()-tick,
                peak_allocated_bytes=torch.cuda.max_memory_allocated())
        except torch.cuda.OutOfMemoryError as error:
            benchmark.update(status='oom',error=str(error))
            torch.cuda.empty_cache()
        benchmarks.append(benchmark)
        print('dev benchmark',json.dumps(benchmark),flush=True)
        result['stages']['dev_batch_benchmark']=benchmarks;write('result.json',result)
    successful=[b['batch_size'] for b in benchmarks if b['status']=='passed']
    assert successful,'No dev batch fits'
    for batch_size in reversed(successful):
        torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats()
        try:
            with torch.inference_mode():
                batch=dev_prompt.repeat(batch_size,1)
                generated=model.generate(input_ids=batch,attention_mask=torch.ones_like(batch),do_sample=False,
                    min_new_tokens=MAX_NEW_TOKENS,max_new_tokens=MAX_NEW_TOKENS,use_cache=True,
                    pad_token_id=tokenizer.pad_token_id,eos_token_id=eos,use_model_defaults=False)
            assert generated.shape[1]-batch.shape[1]==MAX_NEW_TOKENS
            dev_peak=torch.cuda.max_memory_allocated()
            result['stages']['longest_dev']=dict(batch_size=batch_size,prompt_tokens=dev_prompt.numel(),
                forced_generated_tokens=MAX_NEW_TOKENS,peak_allocated_bytes=dev_peak)
            del generated
            break
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
    else:raise RuntimeError('No dev batch fits full completion budget')
    assert adapter_digest(model,'policy')==saved_digest and adapter_digest(model,'reference')==reference_digest
    result.update(status='completed',wall_seconds=time.monotonic()-started,production_runs_modified=False)
    write('result.json',result);print('All GPU acceptance checks passed; verification model exits',flush=True)


if __name__=='__main__':
    try:main()
    except BaseException as error:
        if OUT.exists():write('failure.json',dict(type=type(error).__name__,message=str(error),updated_unix=time.time()))
        raise
