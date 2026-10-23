"""Numerical contract, exhaustive matching oracle, and raw-span checks for R0-v2."""
import itertools
import json
from pathlib import Path
import random
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from action_reward_v2 import reward_action_r0_v2


def event(kind='MDS', trigger='uses', args=(('BaseComponent', 'BERT'),)):
    return dict(event_type=kind, trigger=dict(tokens=trigger.split()),
                arguments=[dict(argument_type=r, tokens=t.split()) for r, t in args])


def actions(events):
    lines = []
    for e in events:
        lines += [f"NEW({e['event_type']})", 'TRG(' + json.dumps(' '.join(e['trigger']['tokens'])) + ')']
        lines += ['ARG(' + a['argument_type'] + ', ' + json.dumps(' '.join(a['tokens'])) + ')' for a in e['arguments']]
        lines += ['END']
    return '\n'.join(lines + ['STOP'])


def score(pred, gold):
    return reward_action_r0_v2(actions(pred), dict(target_events=gold), schema={'MDS': ['BaseComponent']})


@pytest.mark.parametrize('pred,expected', [
    ([event()], 1), ([event('WKS')], .75), ([event(args=[('Content', 'BERT')])], .75),
    ([event('WKS', args=[('Content', 'BERT')])], .5),
    ([event(args=[('BaseComponent', 'wrong')])], .5), ([event(trigger='wrong')], 0),
    ([event(args=[])], .5), ([event('WKS', args=[])], .25),
    ([event(args=[('BaseComponent', 'BERT')] * 2)], 5 / 6), ([event(), event()], 2 / 3),
    ([event('UNKNOWN_TYPE')], .75), ([event(args=[('UNKNOWN_ROLE', 'BERT')])], .75),
    ([], 0),
])
def test_required_numbers(pred, expected):
    result = score(pred, [event()])
    assert result['reward'] == pytest.approx(expected)
    assert result['syntax_valid']
    assert result['reward_version'] == 'r0_v2'


@pytest.mark.parametrize('pred,gold,expected,case', [
    ([], [], 1, 'both_empty'), ([event(args=[])], [event(args=[])], 1, 'head_only'),
    ([event('WKS', args=[])], [event(args=[])], .5, 'head_only'),
    ([event()], [event(args=[])], .5, 'ordinary'), ([event()], [], 0, 'ordinary'),
    ([], [event(args=[])], 0, 'head_only'),
])
def test_empty_cases(pred, gold, expected, case):
    result = score(pred, gold)
    assert result['reward'] == expected and result['reward_case'] == case
    if case in ('head_only', 'both_empty'):
        assert result['components']['argument'] is result['components']['role'] is None
    if case == 'both_empty':
        assert all(v is None for v in result['components'].values())


def test_two_arguments_type_and_one_role_wrong():
    gold = event(trigger='leverages', args=[('BaseComponent', 'adversarial domain adaptation framework'), ('Target', 'introduce')])
    pred = event('WKS', trigger='leverages', args=[('Content', 'adversarial domain adaptation framework'), ('Target', 'introduce')])
    result = score([pred], [gold])
    assert list(result['components'].values()) == [1, 0, 1, .5]
    assert result['reward'] == .625
    assert result['event_hits'] == [dict(trigger=True, type=False)]
    assert result['argument_hits'] == [[dict(argument=True, role=False), dict(argument=True, role=True)]]


@pytest.mark.parametrize('raw', ['', 'garbage', 'STOP\nSTOP', 'STOP\n', 'NEW(MDS)\nSTOP',
    'NEW(MDS)\nTRG("x")\nSTOP', 'NEW(MDS)\nTRG("x")\nEND',
    'NEW(MDS)\nTRG("")\nEND\nSTOP', 'NEW(MDS)\nTRG("x")\nARG(R, " ")\nEND\nSTOP',
    'NEW(MDS)\nTRG("x")\nNEW(MDS)\nSTOP', 'NEW(MDS)\nTRG("x") junk\nEND\nSTOP'])
def test_syntax_invalid_has_no_trace(raw):
    if raw == 'STOP\n':  # Existing splitlines grammar allows one final newline.
        assert reward_action_r0_v2(raw, dict(target_events=[]))['syntax_valid']
        return
    result = reward_action_r0_v2(raw, dict(target_events=[]))
    assert result['reward'] == 0 and not result['syntax_valid']
    assert result['fields'] == [] and result['event_pairs'] is None
    assert all(v is None for v in result['components'].values())


def test_raw_spans_escape_whitespace_duplicates_and_no_eos():
    raw = '  NEW(UNKNOWN_TYPE)  \n TRG("a\\\"b") \n ARG(UNKNOWN_ROLE, "x\\ny")\n ARG(R, "x\\ny")\n END\n STOP'
    gold = [event('UNKNOWN_TYPE', 'a"b', [('UNKNOWN_ROLE', 'x y'), ('R', 'x y')])]
    result = reward_action_r0_v2(raw, dict(target_events=gold), truncated=True, has_eos=False)
    assert result['reward'] == 1 and result['truncated'] and not result['has_eos']
    assert [raw[f['start']:f['end']] for f in result['fields']] == ['UNKNOWN_TYPE', 'a\\"b', 'UNKNOWN_ROLE', 'x\\ny', 'R', 'x\\ny']
    assert all(f['hit'] for f in result['fields'])
    assert [f['argument_index'] for f in result['fields']] == [None, None, 0, 0, 1, 1]


def test_global_denominators_reverse_optimal_pairing():
    gold = [event('A', 't', [('R', 'a')]), event('B', 't', [('R', 'b')])]
    pred = [event('A', 't', [('R', 'b')]), event('B', 't', [('R', 'a')])]
    first = score(pred, gold)
    assert first['reward'] == .75 and first['event_pairs'] == [(0, 1), (1, 0)]
    pred[0]['arguments'] += [dict(argument_type='R', tokens=[f'x{i}']) for i in range(6)]
    second = score(pred, gold)
    assert second['reward'] == .5 and second['event_pairs'] == [(0, 0), (1, 1)]


def test_role_matching_duplicates_and_no_parent_borrowing():
    e = event(args=[('R1', 'x'), ('R2', 'x'), ('R1', 'x')])
    p = event(args=[('R2', 'x'), ('R1', 'x')])
    result = score([p], [e])
    assert result['counts']['matched_role_count'] == 2
    assert result['argument_pairs'] == [(0, 0, 0, 1), (0, 1, 0, 0)]
    gold = [event(trigger='one', args=[('R', 'a')]), event(trigger='two', args=[('R', 'b')])]
    pred = [event(trigger='one', args=[('R', 'b')]), event(trigger='two', args=[('R', 'a')])]
    assert score(pred, gold)['reward'] == .5
    assert score(pred[:1], gold)['reward'] == pytest.approx(1 / 3)
    assert score(gold[:1], gold)['reward'] == pytest.approx(2 / 3)


def brute_score(pred, gold):
    """Independent exhaustive partial bijections, including argument permutations."""
    de = len(pred) + len(gold)
    da = sum(len(e['arguments']) for e in pred + gold)
    if not de:
        return 1
    def argument_best(p, g):
        pa, ga = p['arguments'], g['arguments']
        best = 0
        for assignment in itertools.product(range(-1, len(ga)), repeat=len(pa)):
            used = [i for i in assignment if i >= 0]
            if len(used) != len(set(used)):
                continue
            value = 0
            for pi, gi in enumerate(assignment):
                if gi >= 0:
                    if pa[pi]['tokens'] != ga[gi]['tokens']:
                        break
                    value += 1 + (pa[pi]['argument_type'] == ga[gi]['argument_type'])
            else:
                best = max(best, value)
        return best
    best = 0
    for assignment in itertools.product(range(-1, len(gold)), repeat=len(pred)):
        used = [i for i in assignment if i >= 0]
        if len(used) != len(set(used)):
            continue
        value = 0
        for pi, gi in enumerate(assignment):
            if gi >= 0:
                p, g = pred[pi], gold[gi]
                if p['trigger']['tokens'] != g['trigger']['tokens']:
                    break
                heads = 1 + (p['event_type'] == g['event_type'])
                value += heads / (2 * de) + argument_best(p, g) / (2 * da) if da else heads / de
        else:
            best = max(best, value)
    return best


def test_random_exhaustive_oracle_trace_capacity_and_order():
    rng = random.Random(42)
    for _ in range(65):
        pred, gold = [[event(rng.choice(['A', 'B']), rng.choice(['t', 'u']),
                            [(rng.choice(['R1', 'R2']), rng.choice(['a', 'b'])) for _ in range(rng.randrange(3))])
                       for _ in range(rng.randrange(4))] for _ in range(2)]
        result = score(pred, gold)
        assert result['reward'] == pytest.approx(brute_score(pred, gold))
        assert result == score(pred, gold)
        assert result['reward'] == pytest.approx(score(list(reversed(pred)), list(reversed(gold)))['reward'])
        c = result['counts']
        assert c['matched_type_count'] <= c['matched_trigger_count'] <= min(len(pred), len(gold))
        assert c['matched_role_count'] <= c['matched_argument_count'] <= min(c['pred_argument_count'], c['gold_argument_count'])
        for name in ('trigger', 'type'):
            assert sum(h[name] for h in result['event_hits']) == c[f'matched_{name}_count']
        for name in ('argument', 'role'):
            assert sum(h[name] for es in result['argument_hits'] for h in es) == c[f'matched_{name}_count']


def test_gold_tokens_take_precedence_and_bad_gold_raises_even_if_prediction_invalid():
    gold = event()
    gold['trigger']['text'] = 'wrong fallback'
    gold['arguments'][0]['text'] = 'wrong fallback'
    assert score([event()], [gold])['reward'] == 1
    for malformed in [dict(event_type='MDS', trigger=dict(tokens=[]), arguments=[]),
                      dict(event_type='MDS', trigger=dict(tokens=['x']), arguments=[dict(argument_type='R')])]:
        with pytest.raises((ValueError, KeyError, TypeError)):
            reward_action_r0_v2('broken', dict(target_events=[malformed]))


def test_extra_duplicate_instance_fields_are_unmatched_not_globally_protected():
    result = score([event(), event()], [event()])
    assert sum(h['trigger'] for h in result['event_hits']) == 1
    assert sum(h['type'] for h in result['event_hits']) == 1
    assert sum(f['hit'] for f in result['fields']) == 4
    assert len(result['fields']) == 8
    result = score([event(args=[('BaseComponent', 'BERT')] * 2)], [event()])
    assert sum(h['argument'] for h in result['argument_hits'][0]) == 1
    assert sum(f['hit'] for f in result['fields']) == 4


def test_source_absence_no_filter_and_ordered_tokens_not_normalized():
    row = dict(target_events=[event()], target_sentence='unrelated', tokens=['unrelated'])
    assert reward_action_r0_v2(actions([event()]), row)['reward'] == 1
    assert score([event(trigger='Uses')], [event()])['reward'] == 0
    assert score([event(trigger='a b')], [event(trigger='b a')])['reward'] == 0
    assert score([event(args=[('BaseComponent', 'a b')])], [event(args=[('BaseComponent', 'b a')])])['reward'] == .5


@pytest.mark.parametrize('raw', ['STOP\n\n', 'NEW(BAD-TYPE)\nTRG("x")\nEND\nSTOP',
    'NEW(MDS)\nTRG("x")\nARG(BAD-ROLE, "x")\nEND\nSTOP',
    'NEW(MDS)\nTRG("bad\\q")\nEND\nSTOP',
    'NEW(MDS)\nTRG("unterminated)\nEND\nSTOP',
    'NEW(MDS)\nTRG("x")\nARG(R, 1)\nEND\nSTOP'])
def test_true_syntax_errors_not_unknown_labels(raw):
    result = reward_action_r0_v2(raw, dict(target_events=[event()]))
    assert not result['syntax_valid'] and result['reward'] == 0
    assert result['fields'] == [] and result['argument_hits'] is None


def test_trace_json_serializable_and_gold_unchanged():
    gold = [event(), event(args=[('R1', 'x'), ('R2', 'x')])]
    before = json.dumps(gold)
    result = score(list(reversed(gold)), gold)
    assert json.loads(json.dumps(result))['reward'] == 1
    assert json.dumps(gold) == before


def test_normalized_structure_hash_ignores_instance_order_not_multiplicity():
    gold = [event(), event(args=[('R1', 'a b'), ('R2', 'x')])]
    pred = [dict(e, arguments=list(reversed(e['arguments']))) for e in reversed(gold)]
    first, second = score(gold, gold), score(pred, gold)
    assert first['normalized_structure'] == second['normalized_structure']
    assert hash(first['normalized_structure']) == hash(second['normalized_structure'])
    assert score(gold + gold[:1], gold)['normalized_structure'] != first['normalized_structure']
    assert reward_action_r0_v2('STOP\nEND', dict(target_events=[]))['normalized_structure'] is None


def test_gold_error_not_hidden_by_missing_stop():
    with pytest.raises(ValueError, match='Malformed Gold mention'):
        reward_action_r0_v2('NEW(MDS)\nTRG("x")\nEND',
                            dict(target_events=[dict(event_type='MDS', trigger=dict(tokens=[]), arguments=[])]))


def test_escaped_json_content_spans_are_raw_and_literal_newline_is_invalid():
    raw = 'NEW(MDS)\r\nTRG("\\u0075ses")\r\nARG(BaseComponent, "BE\\u0052T")\r\nEND\r\nSTOP\r\n'
    result = reward_action_r0_v2(raw, dict(target_events=[event()]))
    assert result['reward'] == 1
    assert [raw[f['start']:f['end']] for f in result['fields']] == ['MDS', '\\u0075ses', 'BaseComponent', 'BE\\u0052T']
    invalid = 'NEW(MDS)\nTRG("us\nes")\nEND\nSTOP'
    assert not reward_action_r0_v2(invalid, dict(target_events=[]))['syntax_valid']
