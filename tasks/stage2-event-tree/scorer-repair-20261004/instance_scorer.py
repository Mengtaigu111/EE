"""Instance-preserving SciEvents evaluator; independent of official utils.py.

TI matches the trigger-expression multiset; TC matches (type, expression).
Within each TC group, events are assigned one-to-one by maximum total AC,
then AI, then EC_C, then EC_I. This explicit priority favors correct roles
over additional untyped matches. Each argument metric uses its own multiset
intersection within the selected event pair: an occurrence cannot be reused
within a metric, while typed and untyped metrics are evaluated independently.
EC keeps the official relation definition: an argument is a subevent iff its
expression occurs among that document's triggers. EC TP additionally requires
membership on both sides. Thus token mode cannot recover positional identity
for same-text mentions; it does not assert structural event-edge correctness.

Event and argument signatures are canonically sorted before assignment, and
SciPy's deterministic assignment breaks remaining equal-weight ties. Identical
instances retain multiplicity; their index-level pairing is interchangeable.
Offset sequences are ordered tuples (including reversed/discontinuous spans).
Token expressions are exact tuples of tokens, with text.split() used only if
tokens are absent. Scores use fractions [0, 1], matching official scale.
"""
from collections import Counter, defaultdict

import numpy as np
from scipy.optimize import linear_sum_assignment


METRICS = ('TI', 'TC', 'AI', 'AC', 'EC_I', 'EC_C')
METRIC_KEYS = tuple(f'{name}_{suffix}' for suffix in ('f1', 'p', 'r') for name in METRICS)


def _expression(mention, mode):
    if mode == 'offset':
        if 'offsets' not in mention:
            raise ValueError(f'Offset mode requires offsets: {mention!r}')
        return tuple(mention['offsets'])
    if 'tokens' in mention:
        return tuple(mention['tokens'])
    if 'text' in mention:
        return tuple(mention['text'].split())
    raise ValueError(f'Token mode requires tokens or text: {mention!r}')


def _prepare(events, mode):
    prepared = []
    for index, event in enumerate(events):
        trigger = _expression(event['trigger'], mode)
        args = Counter((_expression(arg, mode), arg['argument_type'])
                       for arg in event['arguments'])
        untyped = Counter()
        for (expr, role), number in args.items():
            untyped[expr] += number
        signature = repr((event['event_type'], trigger, sorted(
            ((repr(key), number) for key, number in args.items()))))
        prepared.append({'index': index, 'type': event['event_type'],
                         'trigger': trigger, 'typed': args, 'untyped': untyped,
                         'signature': signature})
    triggers = {e['trigger'] for e in prepared}
    for event in prepared:
        event['ec_typed'] = Counter({key: number for key, number in event['typed'].items()
                                     if key[0] in triggers})
        event['ec_untyped'] = Counter({key: number for key, number in event['untyped'].items()
                                       if key in triggers})
    return prepared


def _overlap(left, right):
    return sum((left & right).values())


def _pair_counts(gold, pred):
    return {'AC': _overlap(gold['typed'], pred['typed']),
            'AI': _overlap(gold['untyped'], pred['untyped']),
            'EC_C': _overlap(gold['ec_typed'], pred['ec_typed']),
            'EC_I': _overlap(gold['ec_untyped'], pred['ec_untyped'])}


def _assign(golds, preds):
    """Exact lexicographic objective encoded in bounded integer mixed radix."""
    golds = sorted(golds, key=lambda e: (e['signature'], e['index']))
    preds = sorted(preds, key=lambda e: (e['signature'], e['index']))
    if not golds or not preds:
        return []
    values = [[_pair_counts(g, p) for p in preds] for g in golds]
    # Every total objective is <= the smaller side's argument count. One unit
    # at a higher priority therefore outweighs every possible lower objective.
    bound = min(sum(sum(e['untyped'].values()) for e in golds),
                sum(sum(e['untyped'].values()) for e in preds))
    base = bound + 1
    weights = [[((v['AC'] * base + v['AI']) * base + v['EC_C']) * base + v['EC_I']
                for v in row] for row in values]
    # SciPy converts costs to double. Reject impractically large groups rather
    # than silently rounding away a lexicographic priority.
    if max(max(row) for row in weights) * min(len(golds), len(preds)) >= 2 ** 53:
        raise ValueError('TC group too large for exact lexicographic assignment')
    rows, cols = linear_sum_assignment(np.asarray(weights, dtype=np.int64), maximize=True)
    return [(golds[int(r)], preds[int(c)], values[int(r)][int(c)])
            for r, c in zip(rows, cols)]


def score_documents(golds, preds, mode='offset'):
    """Score equally ordered document event lists; never mutate either input.

    Returns metrics (18 official-named keys), counts (six tp/fp/fn maps), and
    matching (one dictionary per document with original event indices, selected
    pairs and unmatched indices). Missing required expressions fail explicitly.
    """
    if mode not in ('offset', 'token'):
        raise ValueError(f'Invalid expression mode: {mode!r}')
    if len(golds) != len(preds):
        raise ValueError('Gold and prediction document counts differ')
    totals = {name: {'tp': 0, 'gold': 0, 'pred': 0} for name in METRICS}
    matching = []
    for doc_index, (gold_doc, pred_doc) in enumerate(zip(golds, preds)):
        gs, ps = _prepare(gold_doc, mode), _prepare(pred_doc, mode)
        for side, events in (('gold', gs), ('pred', ps)):
            totals['TI'][side] += len(events)
            totals['TC'][side] += len(events)
            for name, field in (('AI', 'untyped'), ('AC', 'typed'),
                                ('EC_I', 'ec_untyped'), ('EC_C', 'ec_typed')):
                totals[name][side] += sum(sum(e[field].values()) for e in events)
        totals['TI']['tp'] += _overlap(Counter(e['trigger'] for e in gs),
                                       Counter(e['trigger'] for e in ps))
        groups_g, groups_p = defaultdict(list), defaultdict(list)
        for events, groups in ((gs, groups_g), (ps, groups_p)):
            for e in events:
                groups[(e['type'], e['trigger'])].append(e)
        pairs = []
        for key in sorted(groups_g.keys() & groups_p.keys(), key=repr):
            for gold, pred, counts in _assign(groups_g[key], groups_p[key]):
                totals['TC']['tp'] += 1
                for name, tp in counts.items():
                    totals[name]['tp'] += tp
                pairs.append({'gold_event_index': gold['index'],
                              'pred_event_index': pred['index'],
                              'argument_tp': counts})
        matched_g = {pair['gold_event_index'] for pair in pairs}
        matched_p = {pair['pred_event_index'] for pair in pairs}
        matching.append({'document_index': doc_index, 'pairs': pairs,
                         'unmatched_gold_event_indices': [i for i in range(len(gs)) if i not in matched_g],
                         'unmatched_pred_event_indices': [i for i in range(len(ps)) if i not in matched_p]})
    counts, metrics = {}, {}
    for name, total in totals.items():
        tp = total['tp']
        counts[name] = {'tp': tp, 'fp': total['pred'] - tp, 'fn': total['gold'] - tp}
        p = tp / total['pred'] if total['pred'] else 0.0
        r = tp / total['gold'] if total['gold'] else 0.0
        metrics[name + '_p'] = p
        metrics[name + '_r'] = r
        metrics[name + '_f1'] = 2 * p * r / (p + r) if p + r else 0.0
    return {'metrics': {key: metrics[key] for key in METRIC_KEYS},
            'counts': counts, 'matching': matching}
