"""Single-pass text Action GRPO, using a frozen copy of the selected SFT LoRA."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import random
import shutil
import time


def reference_digest(model):
    import torch
    digest=hashlib.sha256()
    for name,value in sorted(model.named_parameters()):
        if '.reference.' in name:
            digest.update(name.encode());digest.update(value.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def prompt_tokens(tokenizer,row,device):
    text=tokenizer.apply_chat_template([{'role':'user','content':row['prompt']}],tokenize=False,
                                       add_generation_prompt=True,enable_thinking=False)
    return tokenizer(text,add_special_tokens=False,return_tensors='pt')['input_ids'].to(device)


def check_sft_source(adapter,read_json,sha256,smoke):
    metrics_path=adapter.parent/'train_metrics.json'
    metrics=read_json(metrics_path)
    if metrics.get('stage')!='action_sft' or metrics.get('status')!='completed':
        raise ValueError('GRPO requires a completed Action SFT run')
    if metrics.get('smoke') and not smoke:
        raise ValueError('A smoke SFT checkpoint cannot initialize a formal run')
    if metrics.get('best_adapter_sha256')!=sha256(adapter/'adapter_model.safetensors'):
        raise ValueError('Initialization must be the selected best Action SFT adapter')
    if metrics.get('best_adapter_config_sha256')!=sha256(adapter/'adapter_config.json'):
        raise ValueError('Selected Action SFT adapter config changed or unfrozen')
    return metrics_path


def build_parser():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--init-adapter',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--gradient-accumulation',type=int,default=4,help='Prompt groups per optimizer update')
    parser.add_argument('--dev-steps',type=int,default=200)
    parser.add_argument('--smoke',action='store_true',help='8 train rows and first 2 complete dev documents; non-formal')
    parser.add_argument('--reward-version',choices=['event_link_token_exact_v1','r0_v2'],default='event_link_token_exact_v1')
    parser.add_argument('--policy-update-mask',choices=['off','field_v2'],default='off')
    parser.add_argument('--max-steps',type=int,help='Optimizer update budget, at most one train pass')
    parser.add_argument('--early-stopping-patience',type=int,help='Consecutive full dev checks without TC gain >= .001; default disabled')
    return parser


def validate_options(args):
    if args.gradient_accumulation<1 or args.dev_steps<1 or (args.max_steps is not None and args.max_steps<1):
        raise ValueError('Accumulation, dev interval and update budget must be positive')
    if args.policy_update_mask=='field_v2' and args.reward_version!='r0_v2':
        raise ValueError('field_v2 requires r0_v2 reward trace')
    if args.early_stopping_patience is not None and args.early_stopping_patience<1:
        raise ValueError('Early stopping patience must be positive')


class DevEarlyStop:
    """TC plateau anchor independent of lexicographic checkpoint selection."""
    def __init__(self,patience=None,min_delta=.001):
        if patience is not None and (isinstance(patience,bool) or not isinstance(patience,int) or patience<1):
            raise ValueError('Early stopping patience must be a positive integer or None')
        if not math.isfinite(min_delta) or min_delta<=0:
            raise ValueError('Minimum effective TC gain must be positive and finite')
        self.patience=patience;self.min_delta=min_delta;self.anchor_tc=None;self.bad_count=0;self.last_step=-1

    def update(self,step,tc):
        if isinstance(step,bool) or not isinstance(step,int) or step<=self.last_step or step<0:
            raise ValueError('Dev steps must increase, starting at zero or later')
        if not math.isfinite(tc) or not 0<=tc<=1:
            raise ValueError('TC F1 must be finite in [0,1]')
        if self.anchor_tc is None or tc-self.anchor_tc>=self.min_delta-1e-12:
            self.anchor_tc=tc;self.bad_count=0
        else:self.bad_count+=1
        self.last_step=step
        return self.state_dict()

    def state_dict(self):
        return dict(patience=self.patience,min_delta=self.min_delta,anchor_tc=self.anchor_tc,
                    bad_count=self.bad_count,last_step=self.last_step,
                    stop=self.patience is not None and self.bad_count>=self.patience)


def audit_training_prefix(observed,train,stopped,step,accumulation):
    actual=train[:len(observed)]
    ids=[row['id'] for row in observed]
    expected_groups=min(step*accumulation,len(train))
    if (not observed or len(observed)!=expected_groups or ids!=[row['id'] for row in actual]
            or len(set(ids))!=len(ids) or (not stopped and len(actual)!=len(train))):
        raise RuntimeError('Observed GRPO data differs from the completed-update train prefix')
    return actual


def limit_training_rows(rows,max_steps,accumulation):
    if max_steps is None:return rows
    if max_steps<1 or max_steps>math.ceil(len(rows)/accumulation):
        raise ValueError('Update budget must fit one train pass without repeating data')
    return rows[:max_steps*accumulation]


def reward_component_summary(rewards):
    result={}
    for field in ('trigger','type','argument','role'):
        values=[r['components'][field] for r in rewards if r['components'][field] is not None]
        result[f'reward_{field}_f1']=sum(values)/len(values) if values else None
        result[f'reward_{field}_f1_count']=len(values)
    return result


class TrainingDiagnostics:
    """Token-weighted mask rates and active-only reward component means."""
    def __init__(self,mode):
        self.mode=mode;self.samples=0;self.total_reward=0.;self.fallbacks={};self.mixed=0
        self.totals={sign:[0,0] for sign in (-1,1)}
        self.components={field:[0.,0] for field in ('trigger','type','argument','role')}

    def observe(self,reward,mask_diagnostics):
        self.samples+=1;self.total_reward+=reward['reward']
        for field,value in reward['components'].items():
            if value is not None:
                self.components[field][0]+=value;self.components[field][1]+=1
        if mask_diagnostics is None:return
        reason=mask_diagnostics['fallback_reason']
        self.mixed+=mask_diagnostics['mixed_boundary_token_count']
        if reason:
            self.fallbacks[reason]=self.fallbacks.get(reason,0)+1
        elif mask_diagnostics['advantage_sign'] in self.totals:
            totals=self.totals[mask_diagnostics['advantage_sign']]
            totals[0]+=mask_diagnostics['kept_data_token_count'];totals[1]+=mask_diagnostics['data_token_count']

    def summary(self):
        result=dict(reward_version='r0_v2',policy_update_mask_mode=self.mode,samples=self.samples,
                    reward_r0_v2=self.total_reward/self.samples if self.samples else None,
                    fallback_reason_counts=self.fallbacks,
                    mask_fallback_rate=sum(self.fallbacks.values())/self.samples if self.samples and self.mode!='off' else None,
                    mixed_boundary_token_count=self.mixed)
        for sign,label in ((1,'positive'),(-1,'negative')):
            kept,total=self.totals[sign]
            result.update({f'{label}_data_keep_ratio':kept/total if total else None,
                           f'{label}_data_token_count':total,f'{label}_kept_data_token_count':kept})
        for field,(total,count) in self.components.items():
            result.update({f'reward_{field}_f1':total/count if count else None,f'reward_{field}_f1_count':count})
        return result


def main():
    args=build_parser().parse_args()
    validate_options(args)
    import torch
    from grpo_core import (activate_adapter,completion_logprobs,completion_mask,group_advantages,
                           grpo_loss,initialize_dual_adapters,save_policy,behavior_logprobs)
    from action_common import (DATA_DIR,MODEL_PATH,MAX_NEW_TOKENS,MAX_LENGTH,sha256,read_json,
                               write_json,read_rows,append_jsonl,prepare_run,verify_locks,load_module,ROOT,generation_arguments)
    from action_evaluation import evaluate_model,parse_prediction
    from action_rewards import reward_action_r0 as reward_action,REWARD_VERSION,normalized_structure
    if args.reward_version=='r0_v2':
        from action_reward_v2 import reward_action_r0_v2 as reward_action,MATCHING_VERSION
        REWARD_VERSION='r0_v2'
    if args.policy_update_mask=='field_v2':
        from action_update_mask import build_field_mask
    from grpo_diagnostics import probability_statistics,ProbabilityGuard
    from transformers import AutoModelForCausalLM,AutoTokenizer,GenerationConfig,set_seed
    if 'TMUX' not in os.environ or not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError('Run GPU training inside tmux on a bf16 CUDA device')
    source_metrics=check_sft_source(args.init_adapter,read_json,sha256,args.smoke)
    probability_guards={name:ProbabilityGuard() for name in ('current_vs_old','cached_vs_eval','eval_vs_train')}
    configuration=dict(probability_guard=probability_guards['current_vs_old'].configuration(),
        reward_version=REWARD_VERSION,reward_formula='0.5 event_exact_f1 + 0.5 link_exact_f1',
        generation_mode='original sequential G4',event_reward_enabled=True,stage='action_grpo',seed=42,group_size=4,gradient_accumulation=args.gradient_accumulation,
        learning_rate=1e-5,clip=.2,beta=.01,temperature=1.,top_p=1.,top_k=0,legal_action_mask=False,
        max_new_tokens=MAX_NEW_TOKENS,max_length=MAX_LENGTH,num_train_epochs=1,num_iterations=1,
        optimizer='AdamW',scheduler='constant',max_grad_norm=1.,lora_dropout=0.,
        loss_type='grpo',scale_rewards='group_sample_std_plus_1e-4',completion_eos_in_loss=True,
        old_probability_source='actual cached generation scores, sampled tokens',
        output_scores=True,return_dict_in_generate=True,
        truncated_completions='keep all sampled tokens and record truncation; strict parser reward',
        shuffle='seed42 original full train once including all None',dev_steps=args.dev_steps,
        reference='same selected Action SFT LoRA, frozen, shared frozen base',smoke=args.smoke,
        selection='full dev presence TC then TI; earlier checkpoint on ties',
        train_sha256=sha256(DATA_DIR/'train.jsonl'),sft_metrics_sha256=sha256(source_metrics),
        source_sha256={str(Path(__file__)):sha256(Path(__file__)),
                       str(Path(__file__).with_name('grpo_core.py')):sha256(Path(__file__).with_name('grpo_core.py'))})
    configuration.update(policy_update_mask_mode=args.policy_update_mask,max_optimizer_steps=args.max_steps,
        early_stopping_patience=args.early_stopping_patience,early_stopping_min_delta=.001,
        early_stopping_metric='full dev TC plateau anchor; initial dev establishes anchor')
    if REWARD_VERSION=='r0_v2':
        configuration.update(reward_formula='equal active mean of trigger/type/argument/role F1',
            matching_version=MATCHING_VERSION,reward_field_weights=[.25]*4,
            no_argument_rule='mean trigger/type only; both empty reward1 with NA components',
            training_parser='strict action syntax, semantic labels never gate scoring')
    run=prepare_run(args.output,stage='action_grpo',init_adapter=args.init_adapter,
                    configuration=configuration,model_path=MODEL_PATH,allow_smoke=args.smoke)
    set_seed(42);os.environ.setdefault('TOKENIZERS_PARALLELISM','false')
    tokenizer=AutoTokenizer.from_pretrained(str(MODEL_PATH),local_files_only=True)
    if tokenizer.pad_token_id is None: tokenizer.pad_token_id=tokenizer.eos_token_id
    train=read_rows(DATA_DIR/'train.jsonl');dev=read_rows(DATA_DIR/'dev.jsonl')
    random.Random(42).shuffle(train)
    if any(len({r['id'] for r in rows})!=len(rows) for rows in (train,dev)):
        raise ValueError('Duplicate train/dev sample IDs')
    if args.smoke:
        train=train[:8]
        document_ids=list(dict.fromkeys(r['document_id'] for r in dev))[:2]
        dev=[r for r in dev if r['document_id'] in document_ids]
    full_train_count=len(train)
    train=limit_training_rows(train,args.max_steps,args.gradient_accumulation)
    write_json(args.output/'planned-train-order.json',dict(ids=[r['id'] for r in train],
        rows=len(train),none_rows=sum(r['n_events']==0 for r in train),shuffle_seed=42,
        full_train_once=not args.smoke and len(train)==full_train_count,
        optimizer_update_budget=math.ceil(len(train)/args.gradient_accumulation)))
    # No prompt truncation. Leave a full completion budget for every prompt.
    prompt_token_cost=0
    for row in train+dev:
        length=prompt_tokens(tokenizer,row,'cpu').numel()
        if length+MAX_NEW_TOKENS>MAX_LENGTH:
            raise ValueError(f"Prompt plus completion budget exceeds context: {row['id']} ({length})")
    model_utils=load_module('_action_rl_model_utils',ROOT/'tasks/stage2-event-tree/scripts/model_utils.py')
    base=AutoModelForCausalLM.from_pretrained(str(MODEL_PATH),local_files_only=True,
        config=model_utils.load_model_config(MODEL_PATH),
        torch_dtype=torch.bfloat16,attn_implementation='sdpa').to('cuda')
    model=initialize_dual_adapters(base,args.init_adapter)
    model.config.use_cache=False
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
    model.enable_input_require_grads()
    frozen_reference=reference_digest(model)
    optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=1e-5,
                                betas=(.9,.999),eps=1e-8,weight_decay=0.)
    eos=model.generation_config.eos_token_id or tokenizer.eos_token_id
    eos_ids=[eos] if isinstance(eos,int) else list(eos)
    generation=GenerationConfig(do_sample=True,temperature=1.,top_p=1.,top_k=0,
        max_new_tokens=MAX_NEW_TOKENS,eos_token_id=eos,pad_token_id=tokenizer.pad_token_id,
        bos_token_id=tokenizer.bos_token_id,repetition_penalty=1.,use_cache=True,
        output_scores=True,return_dict_in_generate=True)
    started=time.monotonic();step=0;pending=0;best_key=None;best_step=None;active=0;total_tokens=0;sampling_teacher_max_difference=0.
    optimizer.zero_grad(set_to_none=True)
    training_diagnostics=TrainingDiagnostics(args.policy_update_mask)
    early_stopping=DevEarlyStop(args.early_stopping_patience)

    def evaluate_and_save():
        nonlocal best_key,best_step
        activate_adapter(model,'policy');model.eval()
        result=evaluate_model(model,tokenizer,dev,args.output/'dev'/f'step-{step}',
                              max_new_tokens=MAX_NEW_TOKENS,batch_size=4)
        metrics=result['metrics'];key=(metrics['TC_f1'],metrics['TI_f1'])
        decision=early_stopping.update(step,metrics['TC_f1'])
        checkpoint=save_policy(model,tokenizer,args.output/'checkpoints'/f'step-{step}')
        append_jsonl(args.output/'dev-history.jsonl',dict(step=step,**result,adapter=str(checkpoint),early_stopping=decision))
        if best_key is None or key>best_key:
            best_key=key;best_step=step
            selected=args.output/'best-adapter'
            if selected.exists(): shutil.rmtree(selected)
            shutil.copytree(checkpoint,selected)
        model.train();activate_adapter(model,'policy')
        return decision

    evaluate_and_save()
    for group_index,row in enumerate(train):
        model.eval();activate_adapter(model,'policy')
        prompt=prompt_tokens(tokenizer,row,'cuda')
        prompt_token_cost+=4*prompt.numel()
        samples=[];eval_logprobs=[]
        for sample_index in range(4):
            with torch.no_grad():
                generated=model.generate(input_ids=prompt,attention_mask=torch.ones_like(prompt),**generation_arguments(generation))
                ids=generated.sequences[:,prompt.shape[1]:].detach();mask=completion_mask(ids,eos_ids)
                old=behavior_logprobs(generated,prompt.shape[1],mask)
                del generated
                policy_teacher=completion_logprobs(model,prompt,ids,mask).detach()
                sampling_teacher_difference=float((old-policy_teacher).abs()[mask].max())
                sampling_teacher_max_difference=max(sampling_teacher_max_difference,sampling_teacher_difference)
                activate_adapter(model,'reference')
                ref=completion_logprobs(model,prompt,ids,mask).detach()
                activate_adapter(model,'policy')
            if step==0 and not torch.allclose(policy_teacher,ref,atol=1e-5,rtol=1e-5):
                raise RuntimeError('Policy/reference probabilities differ before the first update')
            text=tokenizer.decode(ids[0][mask[0]],skip_special_tokens=True)
            truncated=not any(int(ids[0,mask[0].sum()-1])==e for e in eos_ids)
            reward=reward_action(text,row,truncated=truncated,has_eos=not truncated)
            samples.append((ids,mask,old,ref,reward,text,sampling_teacher_difference))
            eval_logprobs.append(policy_teacher)
            del policy_teacher
        rewards=torch.tensor([s[4]['reward'] for s in samples],device='cuda')
        advantages=group_advantages(rewards);active+=int(rewards.std().item()>0)
        model.train();activate_adapter(model,'policy')
        group_loss=0.;group_kl=0.;clip_fraction=0.
        # Never activate reference between a checkpointed policy forward and backward.
        for sample_index,(ids,mask,old,ref,reward,text,sampling_teacher_difference) in enumerate(samples):
            current=completion_logprobs(model,prompt,ids,mask)
            append_jsonl(args.output/'probability-token-log.jsonl',dict(group_index=group_index,id=row['id'],
                sample_index=sample_index,optimizer_step_before=step,
                prompt_tokens=prompt[0].tolist(),completion_tokens=ids[0].tolist(),completion_mask=mask[0].tolist(),
                old_behavior_logprobs=old[0].float().tolist(),current_update_logprobs=current.detach()[0].float().tolist(),
                no_cache_eval_logprobs=eval_logprobs[sample_index][0].float().tolist()))
            probability_diagnostic=probability_statistics(current,old,mask)
            sampling_vs_eval=probability_statistics(eval_logprobs[sample_index],old,mask)
            eval_vs_train=probability_statistics(current,eval_logprobs[sample_index],mask)
            append_jsonl(args.output/'probability-checks.jsonl',dict(group_index=group_index,id=row['id'],
                sample_index=sample_index,optimizer_step_before=step,
                cached_vs_no_cache_eval=sampling_vs_eval,eval_vs_train_checkpointing=eval_vs_train,
                current_update_vs_old_behavior=probability_diagnostic,
                completion_tokens=ids[0].tolist(),completion_mask=mask[0].tolist(),
                old_behavior_logprobs=old[0].float().tolist(),
                current_update_logprobs=current.detach()[0].float().tolist()))
            window_diagnostics=dict(
                current_vs_old=probability_guards['current_vs_old'].observe(current,old,mask),
                cached_vs_eval=probability_guards['cached_vs_eval'].observe(eval_logprobs[sample_index],old,mask),
                eval_vs_train=probability_guards['eval_vs_train'].observe(current,eval_logprobs[sample_index],mask))
            append_jsonl(args.output/'probability-window-checks.jsonl',dict(group_index=group_index,id=row['id'],
                sample_index=sample_index,optimizer_step_before=step,**window_diagnostics))
            field_mask=None;mask_diagnostics=None
            if args.policy_update_mask=='field_v2':
                field_mask,mask_diagnostics=build_field_mask(tokenizer,ids,mask,text,reward,float(advantages[sample_index]))
            if REWARD_VERSION=='r0_v2':training_diagnostics.observe(reward,mask_diagnostics)
            loss,stats=grpo_loss(current,old,ref,mask,advantages[sample_index:sample_index+1],field_mask=field_mask)
            (loss/(4*args.gradient_accumulation)).backward()
            group_loss+=loss.item()/4;group_kl+=stats['kl']/4;clip_fraction+=stats['clip_fraction']/4
            total_tokens+=int(mask.sum())
            append_jsonl(args.output/'rollouts.jsonl',dict(group_index=group_index,id=row['id'],
                optimizer_step_before=step,sample_index=sample_index,prompt_tokens=prompt[0].tolist(),
                completion_tokens=ids[0].tolist(),completion_mask=mask[0].tolist(),
                old_logprobs=old[0].float().tolist(),actual_behavior_logprobs=old[0].float().tolist(),
                sampling_vs_teacher_max_abs_difference=sampling_teacher_difference,reference_logprobs=ref[0].float().tolist(),
                current_update_logprobs=current.detach()[0].float().tolist(),
                current_vs_behavior_probability_statistics=probability_diagnostic,
                sampling_vs_no_cache_eval_probability_statistics=sampling_vs_eval,
                eval_vs_train_checkpointing_probability_statistics=eval_vs_train,
                output=text,advantage=float(advantages[sample_index]),
                policy_update_mask_mode=args.policy_update_mask,mask_diagnostics=mask_diagnostics,
                field_mask=field_mask[0].tolist() if field_mask is not None else None,**reward))
        pending+=1
        if REWARD_VERSION=='r0_v2':write_json(args.output/'reward-mask-summary.json',training_diagnostics.summary())
        reward_stats={}
        if REWARD_VERSION=='r0_v2':
            reward_stats=reward_component_summary([s[4] for s in samples])
            structures=[s[4]['normalized_structure'] if s[4]['syntax_valid'] else ('INVALID',s[5]) for s in samples]
        else:
            event_scores=torch.tensor([s[4]['event_exact_f1'] for s in samples])
            link_scores=torch.tensor([s[4]['link_exact_f1'] for s in samples])
            reward_stats=dict(mean_event_exact_f1=float(event_scores.mean()),event_exact_f1_variance=float(event_scores.var(correction=1)),
                              mean_link_exact_f1=float(link_scores.mean()),link_exact_f1_variance=float(link_scores.var(correction=1)))
            structures=[normalized_structure(parse_prediction(s[5])[0]) if s[4]['parse_valid'] else ('INVALID',s[5]) for s in samples]
        append_jsonl(args.output/'groups.jsonl',dict(group_index=group_index,id=row['id'],
            reward_version=REWARD_VERSION,
            **reward_stats,
            reward_variance=float(rewards.var(correction=1)),
            distinct_normalized_structures=len(set(structures)),
            valid_distinct_normalized_structures=len({structures[i] for i,s in enumerate(samples) if s[4]['parse_valid']}),
            active_group_fraction=active/(group_index+1),
            loss=group_loss,kl=group_kl,clip_fraction=clip_fraction,active=bool(rewards.std().item()>0),
            mean_reward=float(rewards.mean()),std_reward=float(rewards.std()),sampled_tokens=total_tokens))
        if pending==args.gradient_accumulation or group_index==len(train)-1:
            if pending<args.gradient_accumulation:
                for p in model.parameters():
                    if p.grad is not None:p.grad.mul_(args.gradient_accumulation/pending)
            norm=torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],1.)
            if not torch.isfinite(norm):raise RuntimeError('Nonfinite gradient norm')
            optimizer.step();optimizer.zero_grad(set_to_none=True);step+=1;pending=0
            if step%args.dev_steps==0 or group_index==len(train)-1:
                decision=evaluate_and_save()
                if decision['stop']:break
    observed=read_rows(args.output/'groups.jsonl')
    stopped=early_stopping.state_dict()['stop']
    actual_train=audit_training_prefix(observed,train,stopped,step,args.gradient_accumulation)
    write_json(args.output/'observed-train-order.json',dict(ids=[r['id'] for r in actual_train],
        rows=len(actual_train),none_rows=sum(r['n_events']==0 for r in actual_train),shuffle_seed=42,
        full_train_once=not args.smoke and len(actual_train)==full_train_count,
        optimizer_updates=step,planned_optimizer_update_budget=math.ceil(len(train)/args.gradient_accumulation),
        stop_reason='early_stop' if stopped else 'budget_exhausted'))
    if reference_digest(model)!=frozen_reference:raise RuntimeError('Frozen reference weights changed')
    verify_locks(run['input_sha256'])
    write_json(args.output/'train_metrics.json',dict(stage='action_grpo',status='completed',smoke=args.smoke,
        reward_version=REWARD_VERSION,global_step=step,observed_groups=len(actual_train),generated_sequences=4*len(actual_train),
        active_groups=active,active_group_fraction=active/len(actual_train),sampled_completion_tokens=total_tokens,sampled_prompt_tokens=prompt_token_cost,
        effective_completion_batch=4*args.gradient_accumulation,dev_selection_includes_initial=True,
        loss_completion_tokens=total_tokens,observed_train_ids_verified=True,train_passes=len(actual_train)/full_train_count,
        policy_update_mask_mode=args.policy_update_mask,max_optimizer_steps=args.max_steps,
        early_stopping_patience=args.early_stopping_patience,early_stopping_state=early_stopping.state_dict(),
        stop_reason='early_stop' if stopped else 'budget_exhausted',
        initial_actor_reference_groups_verified=min(args.gradient_accumulation,len(actual_train)),
        train_none_rows=sum(r['n_events']==0 for r in actual_train),train_none_fraction=sum(r['n_events']==0 for r in actual_train)/len(actual_train),
        initial_actor_reference_logprobs_verified=True,
        actual_sampling_vs_teacher_max_abs_difference=sampling_teacher_max_difference,
        old_probability_source="actual cached generation scores, sampled tokens",
        best_step=best_step,best_dev_TC_f1=best_key[0],best_dev_TI_f1=best_key[1],
        best_adapter_sha256=sha256(args.output/'best-adapter/adapter_model.safetensors'),
        best_adapter_config_sha256=sha256(args.output/'best-adapter/adapter_config.json'),
        reference_sha256=frozen_reference,reference_unchanged=True,wall_seconds=time.monotonic()-started))


if __name__=='__main__':
    main()
