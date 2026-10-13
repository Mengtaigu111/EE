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
    from grpo_diagnostics import probability_statistics, assert_probability_consistency
    from train_grpo import prompt_tokens, check_sft_source
    if 'TMUX' not in os.environ or not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError('GPU smoke must run inside tmux on a bf16 CUDA GPU')
    check_sft_source(args.init_adapter, read_json, sha256, False)
    script_path = Path(__file__).resolve()
    script_hash = sha256(script_path)
    configuration = dict(stage='action_grpo', smoke=True, purpose='R0 actual probability/backward validation',
                         reward_version='event_link_token_exact_v1', seed=42, groups=4, group_size=4,
                         gradient_accumulation=4, max_new_tokens=MAX_NEW_TOKENS, learning_rate=1e-5,
                         temperature=1., top_p=1., top_k=0, forced_minimum_tokens=False,
                         source_sha256={str(script_path): script_hash},
                         probability_thresholds=dict(max_p99=.15, max_difference=.5,
                                                     max_outside_clip_fraction=.01))
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
    rows = rows[:4]
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
            if not torch.allclose(teacher[mask], reference[mask], atol=1e-5, rtol=1e-5):
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
                         current_update_vs_behavior=(current, old), policy_vs_reference=(teacher, reference))
            statistics = {}
            for label, (a, b) in pairs.items():
                stats = probability_statistics(a, b, mask)
                statistics[label] = stats
                aggregates[label]['current'].append(a.detach()[mask].float().cpu())
                aggregates[label]['old'].append(b.detach()[mask].float().cpu())
                try:
                    assert_probability_consistency(stats)
                except RuntimeError as error:
                    failures.append(f'{label}, group={group_index}, sample={sample_index}: {error}')
            loss, loss_stats = grpo_loss(current, old, reference, mask, advantages[sample_index:sample_index+1])
            (loss/16).backward()
            group_loss += float(loss.detach())/4
            append_jsonl(args.output/'rollouts.jsonl', dict(group_index=group_index, sample_index=sample_index,
                id=row['id'], prompt_tokens=prompt[0].tolist(), completion_tokens=ids[0].tolist(),
                completion_mask=mask[0].tolist(), actual_behavior_logprobs=old[0].float().tolist(),
                eval_teacher_logprobs=teacher[0].float().tolist(), current_update_logprobs=current.detach()[0].float().tolist(),
                reference_logprobs=reference[0].float().tolist(), probability_diagnostics=statistics,
                output=text, advantage=float(advantages[sample_index]), loss=float(loss.detach()),
                **loss_stats, **reward))
            print(f'group={group_index} sample={sample_index} R0={reward["reward"]:.6f} tokens={int(mask.sum())} '
                  f'update_vs_old={statistics["current_update_vs_behavior"]}', flush=True)
        append_jsonl(args.output/'groups.jsonl', dict(group_index=group_index, id=row['id'], loss=group_loss,
            mean_reward=float(rewards.mean()), std_reward=float(rewards.std()), active=bool(rewards.std()>0),
            mean_event_exact_f1=sum(s[6]['event_exact_f1'] for s in samples)/4,
            mean_link_exact_f1=sum(s[6]['link_exact_f1'] for s in samples)/4))
    parameters = [p for p in model.parameters() if p.requires_grad]
    norm = torch.nn.utils.clip_grad_norm_(parameters, 1.)
    if not torch.isfinite(norm):
        raise RuntimeError('Nonfinite smoke gradient norm')
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
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
    result = dict(status='failed' if failures else 'passed', failures=failures, groups=4, generated_sequences=16,
                  optimizer_updates=1, gradient_norm=float(norm), probability_diagnostics=aggregate_statistics,
                  policy_before=policy_before, policy_after=policy_after, policy_changed=policy_before!=policy_after,
                  reference_before=reference_before, reference_after=reference_after,
                  reference_unchanged=reference_before==reference_after,
                  peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(), wall_seconds=time.monotonic()-started,
                  reward_version='event_link_token_exact_v1', has_forced_minimum_tokens=False,
                  same_parameters_for_all_diagnostic_samples=True)
    write_json(args.output/'result.json', result)
    print(result, flush=True)
    if failures:
        raise RuntimeError('Qwen probability smoke failed; formal RL must not start (see result.json)')


if __name__ == '__main__':
    main()
