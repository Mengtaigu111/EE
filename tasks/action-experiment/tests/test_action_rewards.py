"""R0 matches event instances and argument links without changing presence eval."""
import json
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'tasks/action-experiment'))
from action_rewards import match_event_instances, normalized_structure, reward_action_r0

SCHEMA = {'PRP': ('Content', 'Aim'), 'MDS': ('Content', 'Aim')}


def event(kind='PRP', trigger='propose', links=()):
    return {'event_type': kind, 'trigger': {'tokens': trigger.split()},
            'arguments': [{'argument_type': role, 'tokens': text.split()}
                          for role, text in links]}


def actions(events):
    lines = []
    for e in events:
        lines += [f"NEW({e['event_type']})",
                  f"TRG({json.dumps(' '.join(e['trigger']['tokens']))})"]
        lines += [f"ARG({a['argument_type']}, {json.dumps(' '.join(a['tokens']))})"
                  for a in e['arguments']]
        lines.append('END')
    return '\n'.join(lines + ['STOP'])


def score(pred, gold, **kwargs):
    return reward_action_r0(actions(pred), {'target_events': gold}, schema=SCHEMA, **kwargs)


def test_complex_exact_and_order_invariance():
    gold = [event(links=[('Content', 'a b'), ('Content', 'a b'), ('Aim', 'shared')]),
            event(links=[('Content', 'different'), ('Aim', 'shared')]),
            event('MDS', links=[('Content', 'shared')])]
    pred = list(reversed(gold))
    pred = [dict(e, arguments=list(reversed(e['arguments']))) for e in pred]
    result = score(pred, gold)
    assert result['reward'] == result['event_exact_f1'] == result['link_exact_f1'] == 1
    assert result['matched_event_instances'] == 3
    assert result['matched_argument_links'] == 6
    assert result['reward_version'] == 'event_link_token_exact_v1'
    assert result['parse_valid'] and result['parse_error'] is None


def test_assignment_is_global_not_greedy_and_swaps_do_not_change_it():
    pred = [event(links=[('Content', 'a'), ('Content', 'b')]),
            event(links=[('Content', 'a')])]
    gold = [event(links=[('Content', 'a')]), event(links=[('Content', 'b')])]
    for ps in [pred, list(reversed(pred))]:
        for gs in [gold, list(reversed(gold))]:
            result = score(ps, gs)
            assert result['matched_argument_links'] == 2
            assert result['matched_event_instances'] == 2
            assert result['link_exact_f1'] == pytest.approx(4 / 5)
            assert result['reward'] == pytest.approx(.9)


@pytest.mark.parametrize('links', [[], [('Aim', 'a')], [('Content', 'a'), ('Content', 'wrong')],
                                [('Content', 'a'), ('Content', 'a')]])
def test_argument_deletion_wrong_role_extra_and_duplicate_reduce_link_score(links):
    result = score([event(links=links)], [event(links=[('Content', 'a')])])
    assert result['event_exact_f1'] == 1
    assert result['link_exact_f1'] < 1 and result['reward'] < 1


def test_duplicate_full_event_is_an_extra_instance():
    e = event(links=[('Content', 'a')])
    result = score([e, e], [e])
    assert result['event_exact_f1'] == result['link_exact_f1'] == pytest.approx(2 / 3)
    assert result['pred_event_count'] == result['pred_link_count'] == 2
    assert result['gold_event_count'] == result['gold_link_count'] == 1


def test_rectangular_assignment_both_directions_and_unmatched_links_counted():
    short = [event(links=[('Content', 'b')])]
    long = [event(links=[('Content', 'a')]), event(links=[('Content', 'b')]),
            event(links=[('Content', 'c')])]
    for pred, gold in [(short, long), (long, short)]:
        result = score(pred, gold)
        assert result['matched_event_instances'] == result['matched_argument_links'] == 1
        assert result['reward'] == .5


def test_merge_and_split_lose_parent_instance_credit():
    split = [event(links=[('Content', 'a')]), event(links=[('Content', 'b')])]
    merged = [event(links=[('Content', 'a'), ('Content', 'b')])]
    for pred, gold in [(split, merged), (merged, split)]:
        result = score(pred, gold)
        assert result['event_exact_f1'] == pytest.approx(2 / 3)
        assert result['matched_argument_links'] == 1
        assert result['link_exact_f1'] == .5


def test_links_cannot_borrow_from_different_parent():
    gold = [event(trigger='one', links=[('Content', 'a')]),
            event(trigger='two', links=[('Content', 'b')])]
    pred = [event(trigger='one', links=[('Content', 'b')]),
            event(trigger='two', links=[('Content', 'a')])]
    result = score(pred, gold)
    assert result['event_exact_f1'] == 1 and result['link_exact_f1'] == 0
    assert result['reward'] == .5


def test_ordered_text_tokens_exact_no_offsets_or_bpe():
    pred = [event(trigger='b a', links=[('Content', 'b a')])]
    gold = [event(trigger='a b', links=[('Content', 'a b')])]
    assert score(pred, gold)['reward'] == 0
    gold[0]['trigger'].update(text='incorrect fallback', offsets=[987])
    gold[0]['arguments'][0].update(text='incorrect fallback', offsets=[123])
    assert score([event(trigger='a b', links=[('Content', 'a b')])], gold)['reward'] == 1


def test_no_links_uses_event_f1_not_a_free_bonus():
    result = score([event(), event('MDS')], [event()])
    assert result['event_exact_f1'] == result['link_exact_f1'] == pytest.approx(2 / 3)
    assert score([event('MDS')], [event()])['reward'] == 0


@pytest.mark.parametrize('pred,gold,expected', [([], [], 1), ([], [event()], 0),
                                             ([event()], [], 0)])
def test_empty_boundary(pred, gold, expected):
    assert score(pred, gold)['reward'] == expected


@pytest.mark.parametrize('raw', ['', 'garbage', 'NEW(PRP)\nTRG("x")\nSTOP',
                                 'NEW(PRP)\nTRG("x")\nEND',
                                 'NEW(ZZZ)\nTRG("x")\nEND\nSTOP',
                                 'NEW(PRP)\nTRG("x")\nARG(Wrong, "a")\nEND\nSTOP',
                                 'STOP\nSTOP', 'STOP\nNEW(PRP)'])
def test_invalid_action_never_becomes_valid_empty(raw):
    result = reward_action_r0(raw, {'target_events': []}, schema=SCHEMA, truncated=True)
    assert result['reward'] == 0 and not result['parse_valid']
    assert result['parse_error'] and result['predicted_events'] == 0


def test_complete_stop_without_eos_still_scores_and_does_not_repair_incomplete():
    e = event(links=[('Content', 'a')])
    result = score([e], [e], truncated=True, has_eos=False)
    assert result['reward'] == 1 and result['truncated'] and not result['has_eos']
    assert score([], [], truncated=True, has_eos=False)['reward'] == 1
    bad = reward_action_r0(actions([e]).removesuffix('\nSTOP'), {'target_events': [e]},
                           schema=SCHEMA, truncated=True, has_eos=False)
    assert bad['reward'] == 0 and not bad['parse_valid']


def test_normalized_structure_ignores_order_preserves_multiplicity_and_token_order():
    from action_evaluation import parse_prediction
    gold = [event(links=[('Content', 'a b'), ('Aim', 'c')]), event(links=[('Content', 'd')])]
    parsed, _ = parse_prediction(actions(gold), SCHEMA)
    swapped = [dict(e, arguments=list(reversed(e['arguments']))) for e in reversed(parsed)]
    assert normalized_structure(parsed) == normalized_structure(swapped)
    assert hash(normalized_structure(parsed)) == hash(normalized_structure(swapped))
    assert normalized_structure(parsed) != normalized_structure(parsed + [parsed[0]])
    duplicate_link = [dict(parsed[0], arguments=parsed[0]['arguments'] + [parsed[0]['arguments'][0]]), parsed[1]]
    assert normalized_structure(parsed) != normalized_structure(duplicate_link)
    reverse_tokens = [dict(parsed[0], trigger='propose reversed'), parsed[1]]
    assert normalized_structure(parsed) != normalized_structure(reverse_tokens)


def test_default_schema_real_shared_trigger_types_and_cached_parser():
    from action_evaluation import schema_from
    schema_from.cache_clear()
    gold = [event(links=[('Content', 'our model')]), event('MDS')]
    first = reward_action_r0(actions(gold), {'target_events': gold})
    second = reward_action_r0(actions(gold), {'target_events': gold})
    assert first['reward'] == second['reward'] == 1
    assert schema_from.cache_info().misses == 1 and schema_from.cache_info().hits == 1


def test_rectangular_matching_agrees_with_independent_exhaustive_search():
    from collections import Counter
    from itertools import permutations
    import random
    rng = random.Random(42)
    for pred_count, gold_count in [(1, 3), (2, 3), (3, 2), (3, 3)]:
        for _ in range(20):
            pred_links = [[rng.randrange(4) for _ in range(rng.randrange(5))]
                          for _ in range(pred_count)]
            gold_links = [[rng.randrange(4) for _ in range(rng.randrange(5))]
                          for _ in range(gold_count)]
            pred = [event(links=[('Content', str(link)) for link in links])
                    for links in pred_links]
            gold = [event(links=[('Content', str(link)) for link in links])
                    for links in gold_links]
            weights = [[sum((Counter(p) & Counter(g)).values()) for g in gold_links]
                       for p in pred_links]
            if pred_count <= gold_count:
                best = max(sum(weights[p][g] for p, g in enumerate(choice))
                           for choice in permutations(range(gold_count), pred_count))
            else:
                best = max(sum(weights[p][g] for g, p in enumerate(choice))
                           for choice in permutations(range(pred_count), gold_count))
            pairs = match_event_instances(pred, gold)
            assert len(pairs) == min(pred_count, gold_count)
            assert sum(weight for _, _, weight in pairs) == best
            assert len({p for p, _, _ in pairs}) == len(pairs)
            assert len({g for _, g, _ in pairs}) == len(pairs)
