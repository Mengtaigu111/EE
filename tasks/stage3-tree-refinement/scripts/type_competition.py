"""D2 forced-type continuation helpers and scoring primitives."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import re
import time
from typing import Any, Iterable

from common import parse_forest, read_jsonl


_TOKEN = re.compile(r'\(|\)|"(?:[^"\\]|\\.)*"|[^\s()"]+')


def prefix_token_check(greedy_ids: list[int], constructed_ids: list[int],
                       prefix_length: int) -> dict[str, Any]:
    if prefix_length < 1:
        raise ValueError('prefix_length must be positive')
    expected = list(greedy_ids[:prefix_length])
    actual = list(constructed_ids[:prefix_length])
    mismatches = sum(left != right for left, right in zip(expected, actual))
    mismatches += abs(len(expected) - len(actual))
    rate = mismatches / len(expected) if expected else 1.0
    return {'mismatches': mismatches, 'rate': rate, 'use_token_slice': rate > 0.01}


def _escape(text: str) -> str:
    return text.replace('\\', '\\\\').replace('"', '\\"')


def build_forced_prefix(prompt: str, prior_events: Iterable[str], event_type: str,
                        trigger: str) -> str:
    prior = ' '.join(prior_events)
    root_prefix = f'{prompt} (ROOT'
    if prior:
        root_prefix += f' {prior}'
    return f'{root_prefix} ({event_type} (Trigger "{_escape(trigger)}")'


def take_event_subtree(generated: str, max_tokens: int = 160) -> tuple[str, bool]:
    """Keep generated continuation through the first close of the forced event.

    The caller supplies the already-open event prefix, so depth starts at one.
    A false return means the generation hit its explicit budget before closure;
    callers must record/drop it rather than pretend it is a complete tree.
    """
    if max_tokens < 1:
        raise ValueError('max_tokens must be positive')
    tokens = list(_TOKEN.finditer(generated))
    depth = 1
    end = 0
    for token_index, match in enumerate(tokens):
        token = match.group(0)
        if token == '(':
            depth += 1
        elif token == ')':
            depth -= 1
        end = match.end()
        if depth == 0:
            return generated[:end], True
        # Do not cut through a lexical child. Once the budget is reached, keep
        # reading until the current child closes so the caller receives an
        # explicitly incomplete but syntactically inspectable continuation.
        if token_index + 1 >= max_tokens and token == ')':
            return generated[:end], False
    return generated[:end], False


def filter_invalid_roles(event: dict[str, Any], schema: dict[str, set[str]]) -> tuple[dict[str, Any], list[str]]:
    allowed = schema.get(event['type'], set())
    kept, dropped = [], []
    for argument in event.get('args', []):
        if argument['role'] in allowed:
            kept.append(argument)
        else:
            dropped.append(argument['role'])
    result = dict(event)
    result['args'] = kept
    return result, dropped


def candidate_score(logprobs: Iterable[float], forced_prefix_length: int = 0) -> dict[str, float | int]:
    """Aggregate all candidate token log-probabilities, including forced tokens."""
    values = [float(value) for value in logprobs]
    if not values:
        raise ValueError('candidate must contain at least one scored token')
    if forced_prefix_length < 0 or forced_prefix_length > len(values):
        raise ValueError('invalid forced_prefix_length')
    score_sum = math.fsum(values)
    return {'score_sum': score_sum, 'score_mean': score_sum / len(values),
            'token_count': len(values)}


def event_to_sexpr(event: dict[str, Any]) -> str:
    args = ' '.join(f'({argument["role"]} "{_escape(argument["text"])}")'
                    for argument in event.get('args', []))
    trigger = f'(Trigger "{_escape(event["trigger"])}")'
    return f'({event["type"]} {trigger}' + (f' {args}' if args else '') + ')'


def forced_event_prefix(prior_events: Iterable[dict[str, Any]], event_type: str,
                        trigger: str) -> str:
    prior = ' '.join(event_to_sexpr(event) for event in prior_events)
    return f'(ROOT{(" " + prior) if prior else ""} ({event_type} (Trigger "{_escape(trigger)}")'


def parse_forced_candidate(prior_events: Iterable[dict[str, Any]], event_type: str,
                           trigger: str, continuation: str) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    subtree, closed = take_event_subtree(continuation, max_tokens=160)
    if not closed:
        return None, {'closed': False, 'invalid': True}
    raw = forced_event_prefix(prior_events, event_type, trigger) + subtree + ')'
    events, diagnostics = parse_forest(raw)
    matches = [event for event in events if event['type'] == event_type and event['trigger'] == trigger]
    if not matches:
        return None, {'closed': True, **diagnostics, 'invalid': True}
    return matches[-1], {'closed': True, **diagnostics, 'invalid': False}


def sequence_token_logprobs(model: Any, input_ids: Any, score_start: int,
                            score_end: int) -> list[float]:
    """Score target token positions from one teacher-forced sequence."""
    if score_start < 1 or score_end <= score_start or score_end > input_ids.shape[1]:
        raise ValueError('invalid score span')
    outputs = model(input_ids=input_ids)
    logits = outputs.logits[:, :-1, :]
    targets = input_ids[:, 1:]
    log_probs = logits.log_softmax(dim=-1)
    values = log_probs[0, score_start - 1:score_end - 1].gather(
        -1, targets[0, score_start - 1:score_end - 1].unsqueeze(-1)).squeeze(-1)
    return [float(value) for value in values.detach().cpu()]


def _prompt_ids(prompt: str, tokenizer: Any) -> list[int]:
    return tokenizer.apply_chat_template(
        [{'role': 'user', 'content': prompt}], tokenize=True,
        add_generation_prompt=True, enable_thinking=False)


def _find_suffix(sequence: list[int], suffix: list[int]) -> int:
    for index in range(len(sequence) - len(suffix), -1, -1):
        if sequence[index:index + len(suffix)] == suffix:
            return index
    raise ValueError('forced candidate token suffix is not present in prefix')


def _event_candidate_types(event: dict[str, Any], confusion_sets: dict[str, list[str]],
                           split: str, all_types: bool) -> list[str]:
    from ontology import EVENT_TYPES

    if all_types:
        return list(EVENT_TYPES)
    return list(dict.fromkeys([event['type'], *confusion_sets.get(event['type'], [])]))


def generate_type_competitors(row: dict[str, Any], greedy_events: list[dict[str, Any]],
                              candidate_types: list[list[str]], model: Any, tokenizer: Any,
                              device: str = 'cuda', max_new_tokens: int = 160) -> list[dict[str, Any]]:
    """Generate and score D2 subtrees; one batch is used per greedy event."""
    import torch

    results = []
    for event_index, (event, types) in enumerate(zip(greedy_events, candidate_types)):
        prior = greedy_events[:event_index]
        prefixes = [forced_event_prefix(prior, event_type, event['trigger']) for event_type in types]
        encoded = []
        suffixes = []
        for prefix, event_type in zip(prefixes, types):
            ids = _prompt_ids(prefix, tokenizer)
            # The leading space is part of the forced suffix. Qwen's BPE token
            # boundaries differ for `(` versus ` (`; using the exact contextual
            # form is required for prefix verification and likelihood scoring.
            suffix = f' ({event_type} (Trigger "{_escape(event["trigger"])}")'
            suffix_ids = tokenizer(suffix, add_special_tokens=False)['input_ids']
            if suffix_ids and isinstance(suffix_ids[0], list):
                suffix_ids = suffix_ids[0]
            _find_suffix(ids, list(suffix_ids))
            encoded.append({'input_ids': ids})
            suffixes.append(list(suffix_ids))
        batch = tokenizer.pad(encoded, padding=True, return_tensors='pt').to(device)
        with torch.inference_mode():
            generated = model.generate(
                **batch, do_sample=False, max_new_tokens=max_new_tokens,
                pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id,
                return_dict_in_generate=False,
            )
        input_width = batch['input_ids'].shape[1]
        for item_index, (event_type, prefix, suffix_ids) in enumerate(zip(types, prefixes, suffixes)):
            generated_ids = generated[item_index, input_width:]
            cutoff = None
            for count in range(1, len(generated_ids) + 1):
                continuation = tokenizer.decode(generated_ids[:count], skip_special_tokens=True)
                _, closed = take_event_subtree(continuation, max_tokens=max_new_tokens)
                if closed:
                    cutoff = count
                    break
            if cutoff is None:
                results.append({'event_index': event_index, 'type': event_type,
                                'closed': False, 'invalid': True})
                continue
            continuation = tokenizer.decode(generated_ids[:cutoff], skip_special_tokens=True)
            candidate, diagnostics = parse_forced_candidate(prior, event_type, event['trigger'], continuation)
            full_ids = torch.cat([
                batch['input_ids'][item_index, -len(encoded[item_index]['input_ids']):],
                generated_ids[:cutoff],
            ]).unsqueeze(0)
            start = _find_suffix(full_ids[0].tolist(), suffix_ids)
            logprobs = sequence_token_logprobs(model, full_ids, start, full_ids.shape[1])
            scores = candidate_score(logprobs, forced_prefix_length=len(suffix_ids))
            results.append({'event_index': event_index, 'type': event_type,
                            'candidate': candidate, 'continuation': continuation,
                            'diagnostics': diagnostics, 'closed': True,
                            **scores})
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--predictions', type=Path, required=True)
    parser.add_argument('--confusion-sets', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--model', required=True)
    parser.add_argument('--adapter', type=Path, required=True)
    parser.add_argument('--mode', choices=('all', 'confusion'), required=True)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--max-new-tokens', type=int, default=160)
    args = parser.parse_args()
    if 'TMUX' not in os.environ:
        raise RuntimeError('GPU execution must run in tmux')
    if args.output.exists():
        raise FileExistsError(args.output)
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
    from model_utils import load_model_config

    rows = {row['id']: row for row in read_jsonl(args.data)}
    predictions = {row['id']: row for row in read_jsonl(args.predictions)}
    confusion_sets = json.loads(args.confusion_sets.read_text(encoding='utf-8'))
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    tokenizer.padding_side = 'left'
    model = AutoModelForCausalLM.from_pretrained(
        args.model, config=load_model_config(args.model), local_files_only=True,
        torch_dtype=torch.bfloat16, attn_implementation='sdpa').to('cuda')
    model = PeftModel.from_pretrained(model, args.adapter, is_trainable=False).eval()
    set_seed(args.seed)
    started = time.time()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x', encoding='utf-8') as stream:
        for document_id, prediction in predictions.items():
            row = rows[document_id]
            events = prediction['parsed_events']
            candidate_types = [_event_candidate_types(event, confusion_sets, row.get('split', ''), args.mode == 'all')
                               for event in events]
            outputs = generate_type_competitors(row, events, candidate_types, model, tokenizer,
                                                max_new_tokens=args.max_new_tokens)
            stream.write(json.dumps({'id': document_id, 'split': row.get('split'),
                                     'candidates': outputs}, ensure_ascii=False, allow_nan=False) + '\n')
            stream.flush()
    print(json.dumps({'status': 'completed', 'wall_seconds': time.time() - started,
                      'output': str(args.output)}, ensure_ascii=False))


if __name__ == '__main__':
    main()
