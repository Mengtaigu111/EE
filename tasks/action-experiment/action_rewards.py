"""R0 training reward: one-to-one event instances and their exact argument links.

This module reuses the strict Action parser and token conversion. It does not
change the independent text-presence dev/test evaluator.
"""
from collections import Counter, defaultdict

from scipy.optimize import linear_sum_assignment

from action_evaluation import parse_prediction, to_scievents

REWARD_VERSION = 'event_link_token_exact_v1'


def event_key(event):
    """Mention token order is significant; offsets and fallback text are unused."""
    return event['event_type'], tuple(event['trigger']['tokens'])


def argument_links(event):
    """A multiset within one parent event, including repeated identical links."""
    return Counter((arg['argument_type'], tuple(arg['tokens']))
                   for arg in event['arguments'])


def match_event_instances(predicted, gold):
    """Return maximum cardinality exact-key pairs with maximum link overlap.

    Each key forms a complete rectangular candidate graph, so SciPy's rectangular
    assignment always chooses min(pred_count, gold_count) pairs even with zero
    weights. Maximizing the weights then breaks cardinality ties by exact link
    matches. Links from unmatched parents remain in the reward denominator.
    """
    pred_groups, gold_groups = defaultdict(list), defaultdict(list)
    for index, event in enumerate(predicted):
        pred_groups[event_key(event)].append(index)
    for index, event in enumerate(gold):
        gold_groups[event_key(event)].append(index)
    pred_links = [argument_links(e) for e in predicted]
    gold_links = [argument_links(e) for e in gold]
    pairs = []
    for key, pred_indices in pred_groups.items():
        gold_indices = gold_groups.get(key, [])
        if not gold_indices:
            continue
        weights = [[sum((pred_links[p] & gold_links[g]).values())
                    for g in gold_indices] for p in pred_indices]
        pred_positions, gold_positions = linear_sum_assignment(weights, maximize=True)
        pairs.extend((pred_indices[p], gold_indices[g], weights[p][g])
                     for p, g in zip(pred_positions, gold_positions))
    return pairs


def normalized_structure(events):
    """Hashable parsed Action structure, invariant to event/role output order.

    Sorting preserves every repeated event and argument occurrence; it never
    sorts tokens within a mention or merges links across parent instances.
    """
    return tuple(sorted((event['type'], tuple(event['trigger'].split()),
                         tuple(sorted((arg['role'], tuple(arg['text'].split()))
                                      for arg in event['arguments'])))
                        for event in events))


def reward_action_r0(raw_output, row, *, truncated=False, has_eos=None, schema=None):
    """Score complete valid Actions even if generation stopped without EOS.

    Invalid parses receive zero with an explicit error, rather than being
    mistaken for a valid empty prediction. Both denominator counts always
    include unmatched instances. has_eos may be supplied independently of the
    generation-budget diagnostic; by default it is inferred from truncated.
    """
    events, error = parse_prediction(raw_output, schema)
    gold = row['target_events']
    pred = to_scievents(events)
    pred_event_count, gold_event_count = len(pred), len(gold)
    pred_link_count = sum(len(e['arguments']) for e in pred)
    gold_link_count = sum(len(e['arguments']) for e in gold)
    result = dict(reward=0.0, event_exact_f1=0.0, link_exact_f1=0.0,
                  matched_event_instances=0, pred_event_count=pred_event_count,
                  gold_event_count=gold_event_count, matched_argument_links=0,
                  pred_link_count=pred_link_count, gold_link_count=gold_link_count,
                  parse_valid=error is None, parse_error=error,
                  predicted_events=len(events), truncated=bool(truncated),
                  has_eos=not bool(truncated) if has_eos is None else bool(has_eos),
                  reward_version=REWARD_VERSION)
    if error is not None:
        return result
    pairs = match_event_instances(pred, gold)
    matched_events = len(pairs)
    matched_links = sum(weight for _, _, weight in pairs)
    event_denominator = pred_event_count + gold_event_count
    event_f1 = 2 * matched_events / event_denominator if event_denominator else 1.0
    link_denominator = pred_link_count + gold_link_count
    link_f1 = 2 * matched_links / link_denominator if link_denominator else event_f1
    result.update(reward=.5 * event_f1 + .5 * link_f1,
                  event_exact_f1=event_f1, link_exact_f1=link_f1,
                  matched_event_instances=matched_events,
                  matched_argument_links=matched_links)
    return result
