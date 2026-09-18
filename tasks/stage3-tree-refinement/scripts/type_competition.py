"""D2 forced-type continuation helpers and scoring primitives."""

from __future__ import annotations

import math
import re
from typing import Any, Iterable

from common import parse_forest


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
