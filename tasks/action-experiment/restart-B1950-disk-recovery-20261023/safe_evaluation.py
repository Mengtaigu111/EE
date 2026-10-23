"""Same dev protocol, with bounded CUDA-OOM batch fallback and deferred output.

Only generation OOM is retryable. Failed generation stacks are unwound before
cache release; no exception object/traceback or failed GPU tensors are retained.
Frozen action_evaluation parsing, scoring and generation arguments are reused.
"""
from collections import Counter
from pathlib import Path
import time

import torch
from transformers import GenerationConfig
import action_evaluation as _original


def _generate_chunk(model, tokenizer, chunk, config):
    prompts = [tokenizer.apply_chat_template(
        [{'role': 'user', 'content': row['prompt']}], tokenize=False,
        add_generation_prompt=True, enable_thinking=False) for row in chunk]
    batch = tokenizer(prompts, return_tensors='pt', padding=True,
                      add_special_tokens=False, truncation=False)
    device = next(model.parameters()).device
    batch = {key: value.to(device) for key, value in batch.items()}
    try:
        with torch.inference_mode():
            output = model.generate(**batch, **_original.generation_arguments(config))
    except torch.cuda.OutOfMemoryError as error:
        # Return text only. The except target, generate frames and batch references
        # disappear when this function returns, before the caller empty_cache().
        reason = str(error)
        error.__traceback__ = None
        return None, reason
    return output[:, batch['input_ids'].shape[1]:].tolist(), None


def evaluate_model(model, tokenizer, rows, output_dir, *,
                   max_new_tokens=_original.MAX_NEW_TOKENS, batch_size=4):
    if max_new_tokens < 1 or batch_size < 1:
        raise ValueError('Positive generation limits required')
    output_dir = Path(output_dir)
    if output_dir.exists():
        raise FileExistsError(str(output_dir))
    started = time.monotonic()
    was_training = model.training
    old_padding = tokenizer.padding_side
    predictions = []
    retries = []
    batch_row_counts = Counter()
    effective_max_batch_size = min(batch_size, 2)
    active_batch_size = effective_max_batch_size
    try:
        tokenizer.padding_side = 'left'
        model.eval()
        torch.cuda.empty_cache()
        print(f'dev start rows={len(rows)} requested_batch={batch_size} effective_cap={effective_max_batch_size} max_new_tokens={max_new_tokens}', flush=True)
        eos = model.generation_config.eos_token_id or tokenizer.eos_token_id
        eos_ids = {eos} if isinstance(eos, int) else set(eos)
        config = GenerationConfig(do_sample=False, max_new_tokens=max_new_tokens,
                                  eos_token_id=eos, pad_token_id=tokenizer.pad_token_id,
                                  bos_token_id=tokenizer.bos_token_id, use_cache=True)
        start = 0
        while start < len(rows):
            chunk = rows[start:start + active_batch_size]
            completions, oom_reason = _generate_chunk(model, tokenizer, chunk, config)
            if oom_reason is not None:
                failed_size = len(chunk)
                if failed_size == 1:
                    raise torch.cuda.OutOfMemoryError(oom_reason)
                active_batch_size = max(1, active_batch_size // 2)
                retries.append(dict(start_row=start, row_ids=[row['id'] for row in chunk],
                                    failed_batch_size=failed_size,
                                    retry_batch_size=active_batch_size, reason=oom_reason))
                print(f'dev CUDA OOM at row={start}; retry same rows with batch={active_batch_size}', flush=True)
                torch.cuda.empty_cache()
                continue
            if len(completions) != len(chunk):
                raise ValueError('Generation returned an unexpected number of rows')
            for row, tokens in zip(chunk, completions):
                stop = next((i + 1 for i, token in enumerate(tokens) if token in eos_ids), len(tokens))
                completion = tokens[:stop]
                predictions.append(dict(
                    id=row['id'], document_id=row['document_id'], sentence_index=row['sentence_index'],
                    raw_output=tokenizer.decode(completion, skip_special_tokens=True),
                    completion_token_ids=completion,
                    truncated=not any(token in eos_ids for token in completion)))
            batch_row_counts[str(len(chunk))] += len(chunk)
            start += len(chunk)
            if start % 32 == 0 or start == len(rows):
                print(f'dev progress rows={start}/{len(rows)} batch={active_batch_size}', flush=True)
            del completions
    finally:
        tokenizer.padding_side = old_padding
        model.train(was_training)
    # GPU generation or scoring failures leave no prediction/metric files behind.
    result = _original.score_predictions(rows, predictions)
    parsed = result.pop('parsed_predictions')
    result['wall_seconds'] = time.monotonic() - started
    result['max_new_tokens'] = max_new_tokens
    result['adaptive_generation'] = dict(
        policy='Cap dev batch at two; CUDA generation OOM: retry at one, stop at batch-one OOM',
        requested_batch_size=batch_size, effective_max_batch_size=effective_max_batch_size,
        final_batch_size=active_batch_size,
        actual_batch_row_counts=dict(batch_row_counts),
        oom_retry_count=len(retries), oom_retries=retries)
    output_dir.mkdir(parents=True, exist_ok=False)
    for prediction in predictions:
        _original.append_jsonl(output_dir / 'predictions.jsonl', prediction)
    for prediction in parsed:
        _original.append_jsonl(output_dir / 'parsed.jsonl', prediction)
    _original.write_json(output_dir / 'metrics.json', result)
    return result
