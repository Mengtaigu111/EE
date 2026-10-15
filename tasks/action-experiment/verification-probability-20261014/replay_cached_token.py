"""Reconstruct three original AdamW updates and diagnose one cached token.

Diagnostic only: immutable rollout tokens, old/reference probabilities and
advantages are replayed. A teacher mismatch is reported, never hidden. No
generation, reward change, production launch or checkpoint selection occurs.
"""
import argparse
import contextlib
import os
from pathlib import Path
import shutil
import sys
import time

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--old-run', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--teacher-tolerance', type=float, default=1e-5)
    args = parser.parse_args()
    import torch
    from torch.nn.attention import sdpa_kernel, SDPBackend
    from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
    from action_common import (ROOT, MODEL_PATH, prepare_run, read_json, read_rows,
                               write_json, append_jsonl, sha256, load_module, verify_locks)
    from grpo_core import initialize_dual_adapters, activate_adapter, completion_logprobs, grpo_loss
    from train_grpo import check_sft_source, reference_digest
    from grpo_diagnostics import probability_statistics
    if 'TMUX' not in os.environ or not torch.cuda.is_available():
        raise RuntimeError('Run GPU diagnostic inside tmux')
    started = time.monotonic()
    original = read_json(args.old_run/'provenance.json')
    adapter = Path(original['init_adapter'])
    check_sft_source(adapter, read_json, sha256, False)
    # The stopped train entry point/guard can legitimately be amended. Verify
    # original immutable model, adapter, data, reward, codec and evaluator locks.
    changed_sources = {'train_grpo.py', 'grpo_diagnostics.py'}
    immutable = {p: digest for p, digest in original['input_sha256'].items()
                 if Path(p).name not in changed_sources and '/tests/' not in p}
    verify_locks(immutable)
    rollouts = read_rows(args.old_run/'rollouts.jsonl')
    replay = [r for r in rollouts if r['group_index'] < 12]
    expected = [(g, s) for g in range(12) for s in range(4)]
    if [(r['group_index'], r['sample_index']) for r in replay] != expected:
        raise ValueError('Expected exactly the 48 original samples for updates 0–2')
    checks = read_rows(args.old_run/'probability-checks.jsonl')
    target = checks[-1]
    if (target['group_index'], target['sample_index'], target['optimizer_step_before']) != (14, 3, 3):
        raise ValueError('Stopped anomaly identity changed')
    matching = [r for r in rollouts if r['group_index'] == 14]
    if not matching:
        raise ValueError('Missing original prompt tokens for anomaly group')
    target['prompt_tokens'] = matching[0]['prompt_tokens']
    configuration = dict(purpose='same-parameter original cached token replay', smoke=True,
                         old_run=str(args.old_run.resolve()), group_size=4, gradient_accumulation=4,
                         updates=3, teacher_tolerance=args.teacher_tolerance,
                         max_new_tokens=768, learning_rate=1e-5, clip=.2, beta=.01,
                         source_sha256={str(Path(__file__).resolve()): sha256(__file__)})
    run = prepare_run(args.output, stage='action_grpo', init_adapter=adapter,
                      configuration=configuration, allow_smoke=True)
    evidence_paths = [Path(__file__).resolve(), args.old_run/'rollouts.jsonl',
                      args.old_run/'probability-checks.jsonl', args.old_run/'provenance.json']
    for path in evidence_paths:
        run['input_sha256'][str(path.resolve())] = sha256(path)
    script_snapshot = args.output/'source_snapshot'/Path(__file__).resolve().relative_to(ROOT)
    script_snapshot.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(__file__, script_snapshot)
    write_json(args.output/'provenance.json', run)
    set_seed(42)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, local_files_only=True)
    model_utils = load_module('_replay_model_utils', ROOT/'tasks/stage2-event-tree/scripts/model_utils.py')
    base = AutoModelForCausalLM.from_pretrained(MODEL_PATH, local_files_only=True,
        config=model_utils.load_model_config(MODEL_PATH), torch_dtype=torch.bfloat16,
        attn_implementation='sdpa').to('cuda')
    model = initialize_dual_adapters(base, adapter)
    model.config.use_cache = False
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
    model.enable_input_require_grads()
    frozen_reference = reference_digest(model)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                                 lr=1e-5, betas=(.9, .999), eps=1e-8, weight_decay=0.)
    optimizer.zero_grad(set_to_none=True)

    def tensor(values, dtype=torch.long):
        return torch.tensor([values], device='cuda', dtype=dtype)

    replay_max = 0.
    for index, row in enumerate(replay):
        model.train(); activate_adapter(model, 'policy')
        prompt = tensor(row['prompt_tokens'])
        ids = tensor(row['completion_tokens'])
        mask = tensor(row['completion_mask'], torch.bool)
        old = tensor(row['actual_behavior_logprobs'], torch.float32)
        ref = tensor(row['reference_logprobs'], torch.float32)
        current = completion_logprobs(model, prompt, ids, mask)
        recorded = tensor(row['current_update_logprobs'], torch.float32)
        error = float((current.detach()-recorded).abs()[mask].max())
        replay_max = max(replay_max, error)
        advantage = torch.tensor([row['advantage']], device='cuda')
        loss, stats = grpo_loss(current, old, ref, mask, advantage)
        (loss/16).backward()
        append_jsonl(args.output/'update-replay.jsonl', dict(group_index=row['group_index'],
            sample_index=row['sample_index'], teacher_max_abs_difference=error,
            loss=float(loss.detach()), **stats))
        del current, loss
        if (index+1) % 16 == 0:
            norm = torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.)
            if not torch.isfinite(norm):
                raise RuntimeError('Nonfinite reconstructed gradient norm')
            optimizer.step(); optimizer.zero_grad(set_to_none=True)
            print(f'Reconstructed optimizer step {(index+1)//16}, max teacher error {replay_max}', flush=True)
    policy_state = {n: p.detach().cpu().clone() for n, p in model.named_parameters() if '.policy.' in n}
    torch.save(dict(policy=policy_state, optimizer=optimizer.state_dict(), optimizer_step=3,
                    completed_group_cursor=12, includes_partial_groups=False), args.output/'reconstructed-state.pt')
    model.eval(); activate_adapter(model, 'policy')
    prompt = tensor(target['prompt_tokens'])
    ids = tensor(target['completion_tokens'])
    mask = tensor(target['completion_mask'], torch.bool)
    original_teacher = tensor(target['current_update_logprobs'], torch.float32)
    old = tensor(target['old_behavior_logprobs'], torch.float32)
    eos = model.generation_config.eos_token_id or tokenizer.eos_token_id
    eos_ids = [eos] if isinstance(eos, int) else list(eos)

    def selected_probability(logits, label):
        value = logits.float()
        return value.gather(-1, label.reshape(-1, 1)).squeeze(-1)-torch.logsumexp(value, -1)

    def forced_cached():
        values = []
        cache = None
        for position in range(ids.shape[1]):
            input_ids = prompt if position == 0 else ids[:, position-1:position]
            attention = torch.ones((1, prompt.shape[1]+position), device='cuda', dtype=torch.long)
            output = model(input_ids=input_ids, attention_mask=attention, past_key_values=cache, use_cache=True)
            cache = output.past_key_values
            values.append(selected_probability(output.logits[:, -1], ids[:, position]))
            del output
        del cache
        return torch.stack(values, dim=-1)

    def uncached_prefix():
        values = []
        for position in range(ids.shape[1]):
            inputs = torch.cat((prompt, ids[:, :position]), dim=-1)
            output = model(input_ids=inputs, attention_mask=torch.ones_like(inputs), use_cache=False)
            values.append(selected_probability(output.logits[:, -1], ids[:, position]))
            del output
        return torch.stack(values, dim=-1)

    paths = {'original_cached_generation': old, 'original_full_teacher': original_teacher}
    with torch.no_grad():
        paths['reconstructed_sdpa_full'] = completion_logprobs(model, prompt, ids, mask)
        target_error = float((paths['reconstructed_sdpa_full']-original_teacher).abs()[mask].max())
        print(f'Target original teacher match error {target_error}', flush=True)
        paths['reconstructed_sdpa_cached'] = forced_cached()
        paths['reconstructed_sdpa_prefix'] = uncached_prefix()
        with sdpa_kernel(SDPBackend.MATH):
            paths['reconstructed_math_full'] = completion_logprobs(model, prompt, ids, mask)
            paths['reconstructed_math_cached'] = forced_cached()
        # Cast only the final hidden state/projection. Transformer weights and
        # attention remain BF16. Restore the original forward immediately.
        head = model.get_output_embeddings()
        original_forward = head.forward
        def fp32_head(hidden):
            return torch.nn.functional.linear(hidden.float(), head.weight.float(),
                head.bias.float() if head.bias is not None else None)
        head.forward = fp32_head
        try:
            paths['reconstructed_fp32_head_full'] = completion_logprobs(model, prompt, ids, mask)
            paths['reconstructed_fp32_head_cached'] = forced_cached()
        finally:
            head.forward = original_forward
    valid_positions = torch.nonzero(mask[0], as_tuple=False).flatten().tolist()
    rows = []
    for position in valid_positions:
        token = int(ids[0, position])
        probabilities = {name: float(value[0, position]) for name, value in paths.items()}
        rows.append(dict(position_zero_based=position, token_id=token,
                         token=tokenizer.decode([token], skip_special_tokens=False),
                         is_eos=token in eos_ids, logprobs=probabilities,
                         original_ratio=float((original_teacher[0, position]-old[0, position]).exp())))
    write_json(args.output/'tokens.json', rows)
    statistics = {name: probability_statistics(value, old, mask) for name, value in paths.items()}
    pairs = [('reconstructed_sdpa_full', 'reconstructed_sdpa_cached'),
             ('reconstructed_sdpa_prefix', 'reconstructed_sdpa_cached'),
             ('reconstructed_math_full', 'reconstructed_math_cached'),
             ('reconstructed_fp32_head_full', 'reconstructed_fp32_head_cached')]
    pair_statistics = {f'{a}_vs_{b}': probability_statistics(paths[a], paths[b], mask) for a, b in pairs}
    reference_unchanged = reference_digest(model) == frozen_reference
    verify_locks(run['input_sha256'])
    exact = target_error <= args.teacher_tolerance and replay_max <= args.teacher_tolerance
    report = dict(status='completed', exact_reconstruction=exact,
                  target_teacher_max_abs_difference=target_error, replay_teacher_max_abs_difference=replay_max,
                  teacher_tolerance=args.teacher_tolerance, token_count=int(mask.sum()),
                  anomaly=rows[max(valid_positions, key=lambda p: abs(float(original_teacher[0,p]-old[0,p])))],
                  paths_vs_original_cached=statistics, path_pair_statistics=pair_statistics,
                  reference_sha256=frozen_reference, reference_unchanged=reference_unchanged,
                  source_and_adapter_locks_verified=True, wall_seconds=time.monotonic()-started,
                  caveat='Only attribute original anomaly to kernel paths if exact_reconstruction is true; '
                         'FP32-head control does not convert transformer attention/weights to FP32.')
    write_json(args.output/'report.json', report)
    if not reference_unchanged:
        raise RuntimeError('Frozen SFT reference changed')
    print(f'Diagnostic complete; exact reconstruction {exact}; target error {target_error}', flush=True)


if __name__ == '__main__':
    main()
