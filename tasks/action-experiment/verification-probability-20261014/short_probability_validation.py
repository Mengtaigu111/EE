"""Actual Qwen/SFT R0 G=4 probability and backward smoke; never starts formal RL."""
import argparse
import hashlib
import os
from pathlib import Path
import random
import shutil
import sys
import time

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))


def adapter_digest(model, adapter):
    import torch
    digest = hashlib.sha256()
    for name, value in sorted(model.named_parameters()):
        if f'.{adapter}.' in name:
            digest.update(name.encode())
            digest.update(value.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--init-adapter', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig, set_seed
    from action_common import (DATA_DIR, MODEL_PATH, ROOT, MAX_NEW_TOKENS, prepare_run,
                               read_rows, read_json, sha256, write_json, append_jsonl,
                               generation_arguments, load_module, verify_locks)
    from action_rewards import reward_action_r0
    from grpo_core import (initialize_dual_adapters, activate_adapter, completion_mask,
                           behavior_logprobs, completion_logprobs, group_advantages, grpo_loss)
    from grpo_diagnostics import probability_statistics, assert_probability_consistency, ProbabilityGuard
    from train_grpo import prompt_tokens, check_sft_source
    if 'TMUX' not in os.environ or not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError('GPU smoke must run inside tmux on a bf16 CUDA GPU')
    check_sft_source(args.init_adapter, read_json, sha256, False)
    script_path = Path(__file__).resolve()
    script_hash = sha256(script_path)
    configuration = dict(stage='action_grpo', smoke=True, purpose='R0 actual probability/backward validation',
                         reward_version='event_link_token_exact_v1', seed=42, groups=16, group_size=4,
                         gradient_accumulation=4, max_new_tokens=MAX_NEW_TOKENS, learning_rate=1e-5,
                         temperature=1., top_p=1., top_k=0, forced_minimum_tokens=False,
                         source_sha256={str(script_path): script_hash},
                         probability_guard=ProbabilityGuard().configuration())
    run = prepare_run(args.output, stage='action_grpo', init_adapter=args.init_adapter,
                      configuration=configuration, model_path=MODEL_PATH, allow_smoke=True)
    if sha256(script_path) != script_hash:
        raise RuntimeError('Probability smoke source changed during prepare_run')
    run['input_sha256'][str(script_path)] = script_hash
    script_snapshot = args.output/'source_snapshot'/script_path.relative_to(ROOT)
    script_snapshot.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(script_path, script_snapshot)
    if sha256(script_snapshot) != script_hash:
        raise RuntimeError('Probability smoke snapshot hash differs from configuration source hash')
    write_json(args.output/'provenance.json', run)
    verify_locks(run['input_sha256'])
    set_seed(42)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    rows = read_rows(DATA_DIR/'train.jsonl')
    random.Random(42).shuffle(rows)
    rows = rows[:16]
    model_utils = load_module('_r0_smoke_model_utils', ROOT/'tasks/stage2-event-tree/scripts/model_utils.py')
    base = AutoModelForCausalLM.from_pretrained(MODEL_PATH, local_files_only=True,
        config=model_utils.load_model_config(MODEL_PATH), torch_dtype=torch.bfloat16,
        attn_implementation='sdpa').to('cuda')
    model = initialize_dual_adapters(base, args.init_adapter)
    model.config.use_cache = False
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
    model.enable_input_require_grads()
    reference_before = adapter_digest(model, 'reference')
    policy_before = adapter_digest(model, 'policy')
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                                 lr=1e-5, betas=(.9, .999), eps=1e-8, weight_decay=0.)
    eos = model.generation_config.eos_token_id or tokenizer.eos_token_id
    eos_ids = [eos] if isinstance(eos, int) else list(eos)
    generation = GenerationConfig(do_sample=True, temperature=1., top_p=1., top_k=0,
        max_new_tokens=MAX_NEW_TOKENS, eos_token_id=eos, pad_token_id=tokenizer.pad_token_id,
        bos_token_id=tokenizer.bos_token_id, repetition_penalty=1., use_cache=True,
        output_scores=True, return_dict_in_generate=True)
    aggregates = {key: {'current': [], 'old': []} for key in
                  ('cached_vs_eval', 'eval_vs_train_checkpoint', 'current_update_vs_behavior', 'policy_vs_reference')}
    failures = []
    guards={name:ProbabilityGuard() for name in aggregates}
    optimizer.zero_grad(set_to_none=True)
    started = time.monotonic()
    for group_index, row in enumerate(rows):
        activate_adapter(model, 'policy')
        model.eval()
        prompt = prompt_tokens(tokenizer, row, 'cuda')
        samples = []
        for sample_index in range(4):
            with torch.no_grad():
                generated = model.generate(input_ids=prompt, attention_mask=torch.ones_like(prompt),
                                           **generation_arguments(generation))
                ids = generated.sequences[:, prompt.shape[1]:].detach()
                mask = completion_mask(ids, eos_ids)
                old = behavior_logprobs(generated, prompt.shape[1], mask)
                del generated
                teacher = completion_logprobs(model, prompt, ids, mask).detach()
                activate_adapter(model, 'reference')
                reference = completion_logprobs(model, prompt, ids, mask).detach()
                activate_adapter(model, 'policy')
            if group_index<4 and not torch.allclose(teacher[mask], reference[mask], atol=1e-5, rtol=1e-5):
                failures.append(f'Initial policy/reference mismatch group={group_index} sample={sample_index}')
            text = tokenizer.decode(ids[0][mask[0]], skip_special_tokens=True)
            has_eos = int(ids[0, int(mask.sum())-1]) in eos_ids
            reward = reward_action_r0(text, row, truncated=not has_eos, has_eos=has_eos)
            samples.append((ids, mask, old, teacher, reference, text, reward, has_eos))
        rewards = torch.tensor([sample[6]['reward'] for sample in samples], device='cuda')
        advantages = group_advantages(rewards)
        model.train()
        activate_adapter(model, 'policy')
        group_loss = 0.
        for sample_index, (ids, mask, old, teacher, reference, text, reward, has_eos) in enumerate(samples):
            current = completion_logprobs(model, prompt, ids, mask)
            pairs = dict(cached_vs_eval=(teacher, old), eval_vs_train_checkpoint=(current, teacher),
                         current_update_vs_behavior=(current, old))
            if group_index<4:pairs['policy_vs_reference']=(teacher,reference)
            append_jsonl(args.output/'probability-token-log.jsonl',dict(group_index=group_index,sample_index=sample_index,
                prompt_tokens=prompt[0].tolist(),completion_tokens=ids[0].tolist(),completion_mask=mask[0].tolist(),
                old_behavior_logprobs=old[0].float().tolist(),current_update_logprobs=current.detach()[0].float().tolist()))
            statistics = {};window_statistics={}
            for label, (a, b) in pairs.items():
                try:
                    stats = probability_statistics(a, b, mask)
                except (ValueError,RuntimeError) as error:
                    write_json(args.output/'result.json',dict(status='failed',failure=str(error),group_index=group_index,sample_index=sample_index,path=label))
                    raise
                statistics[label] = stats
                aggregates[label]['current'].append(a.detach()[mask].float().cpu())
                aggregates[label]['old'].append(b.detach()[mask].float().cpu())
                try:
                    window_statistics[label]=guards[label].observe(a,b,mask)
                except RuntimeError as error:
                    write_json(args.output/'result.json',dict(status='failed',failure=str(error),group_index=group_index,sample_index=sample_index,path=label,statistics=stats))
                    raise
            if group_index==14 and sample_index==3:
                # Freeze this live model version while replaying the original
                # anomalous response; generation/training modes are restored.
                from torch.nn.attention import sdpa_kernel,SDPBackend
                original_run=HERE/'runs/action-grpo-r0-dev400-42-20261013'
                original_record=read_rows(original_run/'probability-checks.jsonl')[-1]
                fixed_ids=torch.tensor([original_record['completion_tokens']],device='cuda')
                fixed_mask=torch.tensor([original_record['completion_mask']],device='cuda',dtype=torch.bool)
                def fixed_cached():
                    cache=None;values=[]
                    for pos in range(fixed_ids.shape[1]):
                        inputs=prompt if pos==0 else fixed_ids[:,pos-1:pos]
                        attention=torch.ones((1,prompt.shape[1]+pos),device='cuda',dtype=torch.long)
                        output=model(input_ids=inputs,attention_mask=attention,past_key_values=cache,use_cache=True)
                        cache=output.past_key_values;logits=output.logits[:,-1].float()
                        label=fixed_ids[:,pos:pos+1]
                        values.append(logits.gather(-1,label).squeeze(-1)-torch.logsumexp(logits,-1))
                        del output
                    return torch.stack(values,-1)
                model.eval()
                version=adapter_digest(model,'policy')
                with torch.no_grad():
                    full=completion_logprobs(model,prompt,fixed_ids,fixed_mask)
                    cached=fixed_cached()
                    with sdpa_kernel(SDPBackend.MATH):
                        math_full=completion_logprobs(model,prompt,fixed_ids,fixed_mask)
                        math_cached=fixed_cached()
                    head=model.get_output_embeddings();original_forward=head.forward
                    def fp32_head(hidden):
                        return torch.nn.functional.linear(hidden.float(),head.weight.float(),head.bias.float() if head.bias is not None else None)
                    head.forward=fp32_head
                    try:
                        fp_full=completion_logprobs(model,prompt,fixed_ids,fixed_mask)
                        fp_cached=fixed_cached()
                    finally:head.forward=original_forward
                previous=torch.tensor([original_record['current_update_logprobs']],device='cuda')
                comparison=float((full-previous).abs()[fixed_mask].max())
                pos=29;original_cached=float(original_record['old_behavior_logprobs'][pos])
                control_report=dict(parameter_digest=version,parameter_digest_after=adapter_digest(model,'policy'),
                    optimizer_updates_before=3,exact_original_teacher_match=comparison<=1e-5,
                    original_teacher_max_difference=comparison,token_id=int(fixed_ids[0,pos]),token_text=tokenizer.decode([int(fixed_ids[0,pos])]),is_eos=int(fixed_ids[0,pos]) in eos_ids,
                    original_cached_logprob=original_cached,original_teacher_logprob=float(previous[0,pos]),
                    paths={name:dict(token_logprob=float(value[0,pos]),statistics=probability_statistics(value,cached,fixed_mask)) for name,value in [('sdpa_full',full),('sdpa_cached',cached),('math_full',math_full),('math_cached',math_cached),('fp32_head_full',fp_full),('fp32_head_cached',fp_cached)]})
                write_json(args.output/'live-fixed-token-control.json',control_report)
                model.train();activate_adapter(model,'policy')
                del full,cached,math_full,math_cached,fp_full,fp_cached,previous,fixed_ids,fixed_mask
            loss, loss_stats = grpo_loss(current, old, reference, mask, advantages[sample_index:sample_index+1])
            (loss/16).backward()
            group_loss += float(loss.detach())/4
            append_jsonl(args.output/'rollouts.jsonl', dict(group_index=group_index, sample_index=sample_index,
                id=row['id'], prompt_tokens=prompt[0].tolist(), completion_tokens=ids[0].tolist(),
                completion_mask=mask[0].tolist(), actual_behavior_logprobs=old[0].float().tolist(),
                eval_teacher_logprobs=teacher[0].float().tolist(), current_update_logprobs=current.detach()[0].float().tolist(),
                reference_logprobs=reference[0].float().tolist(), probability_diagnostics=statistics,
                probability_window_diagnostics=window_statistics,
                output=text, advantage=float(advantages[sample_index]), loss=float(loss.detach()),
                **loss_stats, **reward))
            print(f'group={group_index} sample={sample_index} R0={reward["reward"]:.6f} tokens={int(mask.sum())} '
                  f'update_vs_old={statistics["current_update_vs_behavior"]}', flush=True)
        append_jsonl(args.output/'groups.jsonl', dict(group_index=group_index, id=row['id'], loss=group_loss,
            mean_reward=float(rewards.mean()), std_reward=float(rewards.std()), active=bool(rewards.std()>0),
            mean_event_exact_f1=sum(s[6]['event_exact_f1'] for s in samples)/4,
            mean_link_exact_f1=sum(s[6]['link_exact_f1'] for s in samples)/4))
        if (group_index+1)%4==0:
            parameters=[p for p in model.parameters() if p.requires_grad]
            norm=torch.nn.utils.clip_grad_norm_(parameters,1.)
            if not torch.isfinite(norm):raise RuntimeError('Nonfinite smoke gradient norm')
            optimizer.step();optimizer.zero_grad(set_to_none=True)
            append_jsonl(args.output/'updates.jsonl',dict(step=(group_index+1)//4,gradient_norm=float(norm)))
    policy_after = adapter_digest(model, 'policy')
    reference_after = adapter_digest(model, 'reference')
    if reference_after != reference_before:
        failures.append('Frozen reference adapter weights changed')
    if policy_before == policy_after:
        failures.append('Policy adapter weights did not change after real R0 update')
    aggregate_statistics = {}
    for label, values in aggregates.items():
        a = torch.cat(values['current'])
        b = torch.cat(values['old'])
        aggregate_statistics[label] = probability_statistics(a, b, torch.ones_like(a, dtype=torch.bool))
        try:
            assert_probability_consistency(aggregate_statistics[label])
        except RuntimeError as error:
            failures.append(f'Aggregate {label}: {error}')
    verify_locks(run['input_sha256'])
    result = dict(status='failed' if failures else 'passed', failures=failures, probability_guard=ProbabilityGuard().configuration(), groups=16, generated_sequences=64,
                  optimizer_updates=4, gradient_norm=float(norm), probability_diagnostics=aggregate_statistics,
                  policy_before=policy_before, policy_after=policy_after, policy_changed=policy_before!=policy_after,
                  reference_before=reference_before, reference_after=reference_after,
                  reference_unchanged=reference_before==reference_after,
                  peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(), wall_seconds=time.monotonic()-started,
                  reward_version='event_link_token_exact_v1', has_forced_minimum_tokens=False,
                  same_parameter_version_within_each_sample=True)
    write_json(args.output/'result.json', result)
    print(result, flush=True)
    if failures:
        raise RuntimeError('Qwen probability smoke failed; formal RL must not start (see result.json)')


if __name__ == '__main__':
    main()
