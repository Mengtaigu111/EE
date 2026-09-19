"""D2 forced-type continuation helpers and scoring primitives.

The forced prefix is placed on the ASSISTANT side of the chat template:

    <chat template of the T0 prompt, with generation prompt>
    (ROOT <greedy events before i> (T (Trigger "t_i")

The model continues from there, so the candidate subtree for type T is
conditioned on exactly the same context the greedy decode saw, plus the forced
type/trigger tokens.  Putting the tree inside the user turn would make the
model start a fresh forest and is not equivalent.
"""

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


def event_to_sexpr(event: dict[str, Any]) -> str:
    args = ' '.join(f'({argument["role"]} "{_escape(argument["text"])}")'
                    for argument in event.get('args', []))
    trigger = f'(Trigger "{_escape(event["trigger"])}")'
    return f'({event["type"]} {trigger}' + (f' {args}' if args else '') + ')'


def forced_suffix(event_type: str, trigger: str) -> str:
    """The forced tokens that open the competing event; leading space included."""
    return f' ({event_type} (Trigger "{_escape(trigger)}")'


def build_prior_text(prior_events: Iterable[str]) -> str:
    """Assistant-side text up to (not including) the competing event."""
    prior = ' '.join(prior_events)
    return '(ROOT' + (f' {prior}' if prior else '')


def build_forced_prefix(prior_events: Iterable[str], event_type: str, trigger: str) -> str:
    """Assistant-side forced prefix text (no prompt, no chat template)."""
    return build_prior_text(prior_events) + forced_suffix(event_type, trigger)


def forced_event_prefix(prior_events: Iterable[dict[str, Any]], event_type: str,
                        trigger: str) -> str:
    return build_forced_prefix([event_to_sexpr(event) for event in prior_events], event_type, trigger)


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


def batch_token_logprobs(model: Any, sequences: list[list[int]], starts: list[int],
                         pad_token_id: int, device: str = 'cuda',
                         batch_size: int = 1) -> list[list[float]]:
    """Teacher-force several right-padded sequences; return per-sequence target log-probs."""
    import torch

    if len(sequences) != len(starts):
        raise ValueError('sequences and starts must have the same length')
    if batch_size < 1:
        raise ValueError('batch_size must be positive')
    result = []
    for offset in range(0, len(sequences), batch_size):
        sequence_chunk = sequences[offset:offset + batch_size]
        start_chunk = starts[offset:offset + batch_size]
        width = max(len(sequence) for sequence in sequence_chunk)
        input_ids = torch.full((len(sequence_chunk), width), pad_token_id, dtype=torch.long)
        attention = torch.zeros((len(sequence_chunk), width), dtype=torch.long)
        for row, sequence in enumerate(sequence_chunk):
            input_ids[row, :len(sequence)] = torch.tensor(sequence, dtype=torch.long)
            attention[row, :len(sequence)] = 1
        input_ids, attention = input_ids.to(device), attention.to(device)
        with torch.inference_mode():
            logits = model(input_ids=input_ids, attention_mask=attention).logits[:, :-1, :].float()
            log_probs = logits.log_softmax(dim=-1)
            gathered = log_probs.gather(-1, input_ids[:, 1:].unsqueeze(-1)).squeeze(-1).cpu()
        for row, (sequence, start) in enumerate(zip(sequence_chunk, start_chunk)):
            if start < 1 or start >= len(sequence):
                raise ValueError('invalid score start')
            result.append([float(value) for value in gathered[row, start - 1:len(sequence) - 1]])
        del input_ids, attention, logits, log_probs, gathered
        if device.startswith('cuda'):
            torch.cuda.empty_cache()
    return result


def _prompt_ids(prompt: str, tokenizer: Any) -> list[int]:
    return tokenizer.apply_chat_template(
        [{'role': 'user', 'content': prompt}], tokenize=True,
        add_generation_prompt=True, enable_thinking=False)


def _text_ids(text: str, tokenizer: Any) -> list[int]:
    ids = tokenizer(text, add_special_tokens=False)['input_ids']
    if ids and isinstance(ids[0], list):
        ids = ids[0]
    return list(ids)


def forced_prefix_ids(prompt: str, prior_text: str, event_type: str, trigger: str,
                      tokenizer: Any) -> tuple[list[int], int, int]:
    """Return (token ids, score_start, forced token count) for one forced prefix.

    ``score_start`` is the index of the first forced token (the ` (T` opening),
    so the forced type and trigger tokens are included in the score span.
    """
    prompt_ids = _prompt_ids(prompt, tokenizer)
    prefix_text = prior_text + forced_suffix(event_type, trigger)
    prefix_ids = _text_ids(prefix_text, tokenizer)
    prior_ids = _text_ids(prior_text, tokenizer)
    if prefix_ids[:len(prior_ids)] == prior_ids:
        forced_count = len(prefix_ids) - len(prior_ids)
    else:
        # BPE merged across the boundary; scoring the whole assistant side is
        # the conservative fallback (constant across candidate types of one event).
        forced_count = len(prefix_ids)
    ids = prompt_ids + prefix_ids
    return ids, len(ids) - forced_count, forced_count


def _find_suffix(sequence: list[int], suffix: list[int]) -> int:
    for index in range(len(sequence) - len(suffix), -1, -1):
        if sequence[index:index + len(suffix)] == suffix:
            return index
    raise ValueError('forced candidate token suffix is not present in prefix')


def greedy_prior_text(raw_output: str, greedy_events: list[dict[str, Any]], index: int) -> tuple[str, bool]:
    """Assistant-side text preceding greedy event ``index``.

    Prefer slicing the stored greedy output so the prior context is exactly what
    the model generated; fall back to re-linearizing parsed events when the
    canonical marker cannot be located in the raw text.
    """
    event = greedy_events[index]
    marker = forced_suffix(event['type'], event['trigger'])
    cursor, hits = 0, 0
    while True:
        position = raw_output.find(marker, cursor)
        if position < 0:
            break
        if hits == sum(1 for prior in greedy_events[:index]
                       if prior['type'] == event['type'] and prior['trigger'] == event['trigger']):
            return raw_output[:position].rstrip(), True
        hits += 1
        cursor = position + 1
    return build_prior_text([event_to_sexpr(item) for item in greedy_events[:index]]), False


def prefix_consistency(row: dict[str, Any], prediction: dict[str, Any],
                       greedy_events: list[dict[str, Any]], tokenizer: Any) -> dict[str, Any]:
    """Compare constructed assistant-side prefixes with the stored greedy tokens.

    For each greedy event, the reconstructed prefix (prior events re-linearized
    plus the forced type/trigger) is tokenized and compared with the tokens of
    the raw greedy output cut at the same point.  A mismatch above one percent
    of the events is reported; the generator then uses the raw-output slice.
    """
    raw = str(prediction.get('raw_output', ''))
    raw_ids = _text_ids(raw, tokenizer)
    checks = []
    for index, event in enumerate(greedy_events):
        marker = forced_suffix(event['type'], event['trigger'])
        prior_text, from_raw = greedy_prior_text(raw, greedy_events, index)
        if not from_raw:
            checks.append({'event_index': index, 'mismatches': 1, 'rate': 1.0,
                           'use_token_slice': True, 'missing_marker': True})
            continue
        greedy_prefix_ids = _text_ids(raw[:len(prior_text)] + marker, tokenizer)
        constructed_ids = _text_ids(
            build_forced_prefix([event_to_sexpr(item) for item in greedy_events[:index]],
                                event['type'], event['trigger']), tokenizer)
        check = prefix_token_check(greedy_prefix_ids, constructed_ids,
                                   max(1, len(greedy_prefix_ids)))
        check['slice_is_prefix_of_greedy_tokens'] = raw_ids[:len(greedy_prefix_ids)] == greedy_prefix_ids
        checks.append({'event_index': index, **check})
    mismatch_count = sum(item['use_token_slice'] for item in checks)
    return {'events': len(checks), 'mismatched_events': mismatch_count,
            'mismatch_rate': mismatch_count / len(checks) if checks else 0.0,
            'checks': checks, 'fallback_required': mismatch_count / len(checks) > 0.01 if checks else False}


def _event_candidate_types(event: dict[str, Any], confusion_sets: dict[str, list[str]],
                           split: str, all_types: bool) -> list[str]:
    from ontology import EVENT_TYPES

    if all_types:
        return list(EVENT_TYPES)
    return list(dict.fromkeys([event['type'], *confusion_sets.get(event['type'], [])]))


def generate_type_competitors(row: dict[str, Any], greedy_events: list[dict[str, Any]],
                              candidate_types: list[list[str]], model: Any, tokenizer: Any,
                              device: str = 'cuda', max_new_tokens: int = 160,
                              raw_output: str | None = None,
                              score_batch_size: int = 2) -> list[dict[str, Any]]:
    """Generate and score D2 subtrees; one generation batch per greedy event."""
    import torch

    results = []
    for event_index, (event, types) in enumerate(zip(greedy_events, candidate_types)):
        prior = greedy_events[:event_index]
        if raw_output is not None:
            prior_text, from_raw = greedy_prior_text(raw_output, greedy_events, event_index)
        else:
            prior_text, from_raw = build_prior_text([event_to_sexpr(item) for item in prior]), False
        encoded, starts, forced_counts = [], [], []
        for event_type in types:
            ids, start, forced_count = forced_prefix_ids(row['prompt'], prior_text, event_type,
                                                         event['trigger'], tokenizer)
            encoded.append({'input_ids': ids})
            starts.append(start)
            forced_counts.append(forced_count)
        if max(len(item['input_ids']) for item in encoded) + max_new_tokens > model.config.max_position_embeddings:
            raise ValueError(f"{row['id']}: forced prefix plus budget exceeds model context")
        batch = tokenizer.pad(encoded, padding=True, return_tensors='pt').to(device)
        with torch.inference_mode():
            generated = model.generate(
                **batch, do_sample=False, max_new_tokens=max_new_tokens,
                pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id,
                temperature=None, top_p=None, top_k=None,
            )
        input_width = batch['input_ids'].shape[1]
        sequences, sequence_starts, pending = [], [], []
        for item_index, event_type in enumerate(types):
            generated_ids = generated[item_index, input_width:].tolist()
            # Stop at EOS/pad; then find the shortest token cut that closes the event.
            for stop in (tokenizer.eos_token_id, tokenizer.pad_token_id):
                if stop in generated_ids:
                    generated_ids = generated_ids[:generated_ids.index(stop)]
            cutoff = None
            for count in range(1, len(generated_ids) + 1):
                continuation = tokenizer.decode(generated_ids[:count], skip_special_tokens=True)
                _, closed = take_event_subtree(continuation, max_tokens=max_new_tokens)
                if closed:
                    cutoff = count
                    break
            if cutoff is None:
                results.append({'event_index': event_index, 'type': event_type,
                                'closed': False, 'invalid': True, 'prior_from_raw': from_raw})
                continue
            continuation = tokenizer.decode(generated_ids[:cutoff], skip_special_tokens=True)
            candidate, diagnostics = parse_forced_candidate(prior, event_type, event['trigger'], continuation)
            sequences.append(encoded[item_index]['input_ids'] + generated_ids[:cutoff])
            sequence_starts.append(starts[item_index])
            pending.append((event_type, candidate, continuation, diagnostics, forced_counts[item_index]))
        if not pending:
            continue
        logprob_rows = batch_token_logprobs(model, sequences, sequence_starts,
                                            tokenizer.pad_token_id, device=device,
                                            batch_size=score_batch_size)
        for (event_type, candidate, continuation, diagnostics, forced_count), logprobs in zip(pending, logprob_rows):
            scores = candidate_score(logprobs, forced_prefix_length=forced_count)
            results.append({'event_index': event_index, 'type': event_type,
                            'candidate': candidate, 'continuation': continuation,
                            'diagnostics': diagnostics, 'closed': True,
                            'prior_from_raw': from_raw, 'forced_tokens': forced_count,
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
    parser.add_argument('--score-batch-size', type=int, default=2,
                        help='teacher-forcing score batch; generation still batches all types per event')
    parser.add_argument('--limit', type=int)
    args = parser.parse_args()
    if 'TMUX' not in os.environ:
        raise RuntimeError('GPU execution must run in tmux')
    if args.output.exists():
        raise FileExistsError(args.output)
    if args.score_batch_size < 1:
        parser.error('--score-batch-size must be positive')
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
    from model_utils import load_model_config

    rows = {row['id']: row for row in read_jsonl(args.data)}
    predictions = {row['id']: row for row in read_jsonl(args.predictions)}
    if set(rows) != set(predictions):
        raise ValueError('data/prediction IDs do not match')
    if args.limit is not None:
        predictions = dict(list(predictions.items())[:args.limit])
    confusion_sets = json.loads(args.confusion_sets.read_text(encoding='utf-8'))
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    tokenizer.padding_side = 'left'
    model = AutoModelForCausalLM.from_pretrained(
        args.model, config=load_model_config(args.model), local_files_only=True,
        torch_dtype=torch.bfloat16, attn_implementation='sdpa').to('cuda')
    model = PeftModel.from_pretrained(model, args.adapter, is_trainable=False).eval()
    model.config.use_cache = True
    set_seed(args.seed)
    started = time.time()
    prefix_audits = []
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x', encoding='utf-8') as stream:
        for done, (document_id, prediction) in enumerate(predictions.items(), 1):
            row = rows[document_id]
            events = prediction['parsed_events']
            audit = prefix_consistency(row, prediction, events, tokenizer)
            prefix_audits.append({k: v for k, v in audit.items() if k != 'checks'})
            candidate_types = [_event_candidate_types(event, confusion_sets, row.get('split', ''), args.mode == 'all')
                               for event in events]
            outputs = generate_type_competitors(row, events, candidate_types, model, tokenizer,
                                                max_new_tokens=args.max_new_tokens,
                                                raw_output=prediction.get('raw_output'),
                                                score_batch_size=args.score_batch_size)
            stream.write(json.dumps({'id': document_id, 'split': row.get('split'), 'mode': args.mode,
                                     'prefix_audit': audit, 'candidates': outputs},
                                    ensure_ascii=False, allow_nan=False) + '\n')
            stream.flush()
            print(json.dumps({'completed': done, 'total': len(predictions),
                              'seconds': round(time.time() - started, 1)}), flush=True)
    summary = {'status': 'completed', 'wall_seconds': time.time() - started,
               'documents': len(predictions), 'mode': args.mode,
               'prefix': {
                   'events': sum(item['events'] for item in prefix_audits),
                   'mismatched_events': sum(item['mismatched_events'] for item in prefix_audits),
                   'fallback_documents': sum(item['fallback_required'] for item in prefix_audits),
               }, 'prefix_audits': prefix_audits}
    print(json.dumps({k: summary[k] for k in ('status', 'wall_seconds', 'prefix')}, ensure_ascii=False))
    args.output.with_suffix('.summary.json').write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


if __name__ == '__main__':
    main()
