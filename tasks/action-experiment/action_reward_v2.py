"""Training-only R0-v2 parser, exact instance matching, and shared field trace.

Formal dev/test parsing remains untouched. Character spans always address the raw
completion, while matching uses decoded strings split with the existing rule.
"""
from collections import defaultdict
import json
import re

from scipy.optimize import linear_sum_assignment

REWARD_VERSION = 'r0_v2'
MATCHING_VERSION = 'trigger_argument_global_f1_v2'
_LABEL = r'[A-Za-z][A-Za-z0-9_]*'
_STRING = r'"(?:[^"\\]|\\.)*"'
_NEW = re.compile(r'NEW\((' + _LABEL + r')\)')
_TRG = re.compile(r'TRG\(\s*(' + _STRING + r')\s*\)')
_ARG = re.compile(r'ARG\((' + _LABEL + r'),\s*(' + _STRING + r')\s*\)')


def _mention(value):
    decoded = json.loads(value)
    if not isinstance(decoded, str) or not decoded.strip():
        raise ValueError('Empty/non-string mention')
    return decoded


def parse_training_actions(raw_output):
    """Parse strict action states without schema filtering; return raw spans.

    Label spelling accepts identifiers, including UNKNOWN_TYPE/UNKNOWN_ROLE.
    Surrounding line whitespace follows the historical parser's strip behavior.
    """
    if not isinstance(raw_output, str) or not raw_output:
        raise ValueError('Empty/non-string actions')
    events, fields = [], []
    state, current, offset = 'between', None, 0
    for number, raw_line in enumerate(raw_output.splitlines(keepends=True), 1):
        line = raw_line.strip()
        base = offset + len(raw_line) - len(raw_line.lstrip())
        offset += len(raw_line)
        if not line or state == 'done':
            raise ValueError(f'Unexpected content at line {number}')
        if line == 'STOP' and state == 'between':
            state = 'done'
            continue
        def field(name, match, group, argument_index=None, quoted=False):
            start, end = match.span(group)
            fields.append(dict(field=name, event_index=len(events),
                               argument_index=argument_index,
                               start=base + start + int(quoted),
                               end=base + end - int(quoted), hit=False))
        match = _NEW.fullmatch(line)
        if match and state == 'between':
            current = dict(event_type=match[1], arguments=[])
            field('type', match, 1)
            state = 'trigger'
            continue
        match = _TRG.fullmatch(line)
        if match and state == 'trigger':
            current['trigger'] = dict(tokens=_mention(match[1]).split())
            field('trigger', match, 1, quoted=True)
            state = 'arguments'
            continue
        match = _ARG.fullmatch(line)
        if match and state == 'arguments':
            index = len(current['arguments'])
            current['arguments'].append(dict(argument_type=match[1], tokens=_mention(match[2]).split()))
            field('role', match, 1, index)
            field('argument', match, 2, index, quoted=True)
            continue
        if line == 'END' and state == 'arguments':
            events.append(current)
            current, state = None, 'between'
            continue
        raise ValueError(f'Illegal action/state at line {number}: {line!r}')
    if state != 'done':
        raise ValueError('Missing END/STOP')
    return events, fields


def _gold_events(events):
    """Validate data independently of prediction validity; never modify Gold."""
    if not isinstance(events, list):
        raise ValueError('Gold target_events must be a list')
    def tokens(mention):
        value = mention.get('tokens')
        if value is None:
            value = mention['text'].split()
        if not isinstance(value, (list, tuple)) or not value or any(not isinstance(t, str) or not t.strip() for t in value):
            raise ValueError(f'Malformed Gold mention tokens: {value!r}')
        return tuple(value)
    output = []
    for e in events:
        if not isinstance(e['event_type'], str) or not e['event_type']:
            raise ValueError('Malformed Gold event type')
        args = []
        if not isinstance(e['arguments'], list):
            raise ValueError('Malformed Gold arguments')
        for a in e['arguments']:
            if not isinstance(a['argument_type'], str) or not a['argument_type']:
                raise ValueError('Malformed Gold argument role')
            args.append(dict(argument_type=a['argument_type'], tokens=tokens(a)))
        output.append(dict(event_type=e['event_type'], trigger=dict(tokens=tokens(e['trigger'])), arguments=args))
    return output


def _argument_pairs(pred, gold):
    """Maximize role matches within exact-text groups, then stable residuals."""
    pg, gg = defaultdict(list), defaultdict(list)
    for i, a in enumerate(pred['arguments']):
        pg[tuple(a['tokens'])].append(i)
    for i, a in enumerate(gold['arguments']):
        gg[tuple(a['tokens'])].append(i)
    pairs = []
    for text, p_indices in pg.items():
        g_indices = gg.get(text, [])
        by_role = defaultdict(list)
        for gi in g_indices:
            by_role[gold['arguments'][gi]['argument_type']].append(gi)
        taken_p, taken_g = set(), set()
        for pi in p_indices:
            available = by_role[pred['arguments'][pi]['argument_type']]
            if available:
                gi = available.pop(0)
                pairs.append((pi, gi))
                taken_p.add(pi)
                taken_g.add(gi)
        pairs.extend(zip((pi for pi in p_indices if pi not in taken_p),
                         (gi for gi in g_indices if gi not in taken_g)))
    return sorted(pairs)


def match_instances(predicted, gold):
    """Maximize exact integer weights with global answer denominators.

    SciPy's deterministic rectangular assignment uses original row/column order
    for tied solutions. No perturbation can change the primary objective.
    """
    de = len(predicted) + len(gold)
    da = sum(len(e['arguments']) for e in predicted + gold)
    pg, gg = defaultdict(list), defaultdict(list)
    for i, e in enumerate(predicted):
        pg[tuple(e['trigger']['tokens'])].append(i)
    for i, e in enumerate(gold):
        gg[tuple(e['trigger']['tokens'])].append(i)
    event_pairs, argument_pairs = [], []
    for text, ps in pg.items():
        gs = gg.get(text, [])
        if not gs:
            continue
        candidates, weights = {}, []
        for pi in ps:
            row = []
            for gi in gs:
                args = _argument_pairs(predicted[pi], gold[gi])
                candidates[pi, gi] = args
                role_hits = sum(predicted[pi]['arguments'][pa]['argument_type'] == gold[gi]['arguments'][ga]['argument_type'] for pa, ga in args)
                heads = 1 + int(predicted[pi]['event_type'] == gold[gi]['event_type'])
                row.append(da * heads + de * (len(args) + role_hits) if da else heads)
            weights.append(row)
        p_positions, g_positions = linear_sum_assignment(weights, maximize=True)
        for pp, gp in zip(p_positions, g_positions):
            pi, gi = ps[int(pp)], gs[int(gp)]
            event_pairs.append((pi, gi))
            argument_pairs.extend((pi, pa, gi, ga) for pa, ga in candidates[pi, gi])
    return sorted(event_pairs), sorted(argument_pairs)


def reward_action_r0_v2(raw_output, row, *, truncated=False, has_eos=None, schema=None):
    """One scalar reward and the exact matching trace shared with field masks.

    schema is accepted for route compatibility and never filters parsed fields.
    Invalid syntax has unavailable trace (None) and no usable field spans.
    """
    gold = _gold_events(row['target_events'])
    names = ('trigger', 'type', 'argument', 'role')
    result = dict(reward_version=REWARD_VERSION, matching_version=MATCHING_VERSION,
                  reward=0.0, syntax_valid=False, parse_valid=False,
                  reward_case='syntax_invalid', components=dict.fromkeys(names),
                  event_pairs=None, argument_pairs=None, event_hits=None,
                  argument_hits=None, fields=[], normalized_structure=None, parse_error=None,
                  predicted_events=0, truncated=bool(truncated),
                  has_eos=not bool(truncated) if has_eos is None else bool(has_eos))
    counts = dict(matched_trigger_count=0, matched_type_count=0,
                  matched_argument_count=0, matched_role_count=0,
                  pred_event_count=0, gold_event_count=len(gold),
                  pred_argument_count=0, gold_argument_count=sum(len(e['arguments']) for e in gold))
    result['counts'] = counts
    try:
        pred, fields = parse_training_actions(raw_output)
    except (ValueError, TypeError) as error:
        result['parse_error'] = str(error)
        result.update(counts)
        return result
    counts.update(pred_event_count=len(pred), pred_argument_count=sum(len(e['arguments']) for e in pred))
    events, args = match_instances(pred, gold)
    event_hits = [dict(trigger=False, type=False) for _ in pred]
    argument_hits = [[dict(argument=False, role=False) for _ in e['arguments']] for e in pred]
    for pi, gi in events:
        event_hits[pi].update(trigger=True, type=pred[pi]['event_type'] == gold[gi]['event_type'])
    for pi, pa, gi, ga in args:
        argument_hits[pi][pa].update(argument=True, role=pred[pi]['arguments'][pa]['argument_type'] == gold[gi]['arguments'][ga]['argument_type'])
    for name in ('trigger', 'type'):
        counts[f'matched_{name}_count'] = sum(h[name] for h in event_hits)
    for name in ('argument', 'role'):
        counts[f'matched_{name}_count'] = sum(h[name] for es in argument_hits for h in es)
    for f in fields:
        hits = event_hits[f['event_index']] if f['argument_index'] is None else argument_hits[f['event_index']][f['argument_index']]
        f['hit'] = hits[f['field']]
    de = counts['pred_event_count'] + counts['gold_event_count']
    da = counts['pred_argument_count'] + counts['gold_argument_count']
    components = result['components']
    case, reward = 'both_empty', 1.0
    if de:
        case = 'ordinary' if da else 'head_only'
        for name in names:
            denominator = de if name in ('trigger', 'type') else da
            components[name] = 2 * counts[f'matched_{name}_count'] / denominator if denominator else None
        active = [v for v in components.values() if v is not None]
        reward = sum(active) / len(active)
    result.update(reward=float(reward), syntax_valid=True, parse_valid=True,
                  reward_case=case, event_pairs=events, argument_pairs=args,
                  event_hits=event_hits, argument_hits=argument_hits,
                  fields=fields, predicted_events=len(pred),
                  normalized_structure=tuple(sorted(
                      (e['event_type'], tuple(e['trigger']['tokens']),
                       tuple(sorted((a['argument_type'], tuple(a['tokens']))
                                    for a in e['arguments']))) for e in pred)))
    result.update(counts)
    return result
