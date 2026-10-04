"""Regression tests for multiplicity-preserving SciEvents scoring (CPU only)."""
import copy
import importlib.util
import itertools
import json
from pathlib import Path
import unittest

from instance_scorer import METRIC_KEYS, score_documents

ROOT = Path(__file__).resolve().parents[3]


def mention(offsets, tokens=None):
    return {'offsets': list(offsets), 'tokens': tokens or [str(x) for x in offsets]}


def event(trigger, args=(), kind='A'):
    return {'event_type': kind, 'trigger': mention(trigger),
            'arguments': [dict(mention(span), argument_type=role) for span, role in args]}


class InstanceScorerTests(unittest.TestCase):
    def assert_count(self, result, metric, tp, fp, fn):
        self.assertEqual(result['counts'][metric], {'tp': tp, 'fp': fp, 'fn': fn})

    def test_self_duplicate_triggers_types_arguments_roles(self):
        doc = [event([1], [([2], 'x'), ([2], 'y'), ([2], 'y')]),
               event([1], [([3], 'z')]), event([1], [([2], 'x')], kind='B'),
               event([2]), event([3])]
        for mode in ('offset', 'token'):
            result = score_documents([doc], [copy.deepcopy(doc)], mode)
            self.assertEqual(set(result['metrics']), set(METRIC_KEYS))
            self.assertTrue(all(x == 1.0 for x in result['metrics'].values()))
            for counts in result['counts'].values():
                self.assertEqual(counts['fp'], 0)
                self.assertEqual(counts['fn'], 0)

    def test_duplicate_prediction_cannot_reuse_gold(self):
        doc = [event([1], [([2], 'x')]), event([2])]
        result = score_documents([doc], [doc + [copy.deepcopy(doc[0])]])
        self.assert_count(result, 'TI', 2, 1, 0)
        self.assert_count(result, 'TC', 2, 1, 0)
        for name in ('AI', 'AC', 'EC_I', 'EC_C'):
            self.assert_count(result, name, 1, 1, 0)
            self.assertEqual(result['metrics'][name + '_p'], .5)

    def test_duplicate_gold_cannot_reuse_prediction(self):
        one = event([1], [([1], 'x')])
        result = score_documents([[one, one]], [[one]])
        for name in result['counts']:
            self.assert_count(result, name, 1, 0, 1)

    def test_same_trigger_different_type_and_best_argument_pair(self):
        gold = [event([1], [([2], 'x')]), event([1], [([3], 'y')]),
                event([1], [([4], 'z')], 'B')]
        pred = [event([1], [([3], 'y')]), event([1], [([2], 'x')]),
                event([1], [([4], 'z')], 'C')]
        result = score_documents([gold], [pred])
        self.assert_count(result, 'TI', 3, 0, 0)
        for name in ('TC', 'AI', 'AC'):
            self.assert_count(result, name, 2, 1, 1)

    def test_argument_roles_use_multisets(self):
        gold = [event([1], [([1], 'x'), ([1], 'x'), ([1], 'y')])]
        pred = [event([1], [([1], 'x'), ([1], 'y'), ([1], 'z')])]
        result = score_documents([gold], [pred])
        for name in ('AI', 'EC_I'):
            self.assert_count(result, name, 3, 0, 0)
        for name in ('AC', 'EC_C'):
            self.assert_count(result, name, 2, 1, 1)

    def test_ac_priority_precedes_ai(self):
        gold = [event([0], [([1], 'x'), ([2], 'y')]),
                event([0], [([3], 'x'), ([4], 'y')])]
        pred = [event([0], [([1], 'x'), ([3], 'z'), ([4], 'z')]),
                event([0], [([3], 'x'), ([1], 'z'), ([2], 'z')])]
        result = score_documents([gold], [pred])
        self.assert_count(result, 'AC', 2, 4, 2)
        self.assert_count(result, 'AI', 2, 4, 2)

    def test_assignment_maximizes_total_not_first_or_greedy_pair(self):
        gold = [event([0], [([1], 'x'), ([2], 'x')]),
                event([0], [([1], 'x')])]
        pred = [event([0], [([1], 'x')]), event([0], [([2], 'x')])]
        result = score_documents([gold], [pred])
        self.assert_count(result, 'AC', 2, 0, 1)
        pairs = result['matching'][0]['pairs']
        self.assertEqual({(p['gold_event_index'], p['pred_event_index']) for p in pairs},
                         {(0, 1), (1, 0)})

    def test_ec_c_breaks_equal_ac_ai_assignment_tie(self):
        gold = [event([0], [([1], 'x'), ([2], 'x')]),
                event([0], [([1], 'y'), ([2], 'y')]), event([1])]
        pred = [event([0], [([1], 'x'), ([2], 'y')]),
                event([0], [([1], 'y'), ([2], 'x')]), event([1])]
        result = score_documents([gold], [pred])
        self.assert_count(result, 'AC', 2, 2, 2)
        self.assert_count(result, 'AI', 4, 0, 0)
        self.assert_count(result, 'EC_C', 2, 0, 0)
        self.assert_count(result, 'EC_I', 2, 0, 0)

    def test_document_boundaries_prevent_cross_document_matching(self):
        result = score_documents([[event([1])], []], [[], [event([1])]])
        self.assert_count(result, 'TI', 0, 1, 1)
        self.assert_count(result, 'TC', 0, 1, 1)

    def test_event_and_argument_permutation_invariance(self):
        gold = [event([0], [([1], 'x'), ([2], 'y')]),
                event([0], [([2], 'y'), ([3], 'z')]), event([2])]
        pred = [event([0], [([2], 'y')]), event([0], [([1], 'z')]), event([2])]
        expected = score_documents([gold], [pred])
        for gs in itertools.permutations(gold):
            for ps in itertools.permutations(pred):
                changed = copy.deepcopy(list(ps))
                for e in changed:
                    e['arguments'].reverse()
                actual = score_documents([list(gs)], [changed])
                self.assertEqual(actual['counts'], expected['counts'])
                self.assertEqual(actual['metrics'], expected['metrics'])

    def test_reverse_offsets_are_not_sorted(self):
        result = score_documents([[event([2, 1])]], [[event([1, 2])]])
        self.assert_count(result, 'TI', 0, 1, 1)

    def test_token_exact_tuple_prefers_tokens_and_falls_back_to_text(self):
        gold = event([1])
        gold['trigger'].update(tokens=['A', 'B'], text='incorrect text')
        pred = event([7])
        pred['trigger'] = {'text': ' A  B '}
        self.assert_count(score_documents([[gold]], [[pred]], 'token'), 'TI', 1, 0, 0)
        pred['trigger'] = {'tokens': ['A B']}
        self.assert_count(score_documents([[gold]], [[pred]], 'token'), 'TI', 0, 1, 1)

    def test_ec_requires_trigger_membership_on_both_sides(self):
        gold = [event([1], [([2], 'x')]), event([2])]
        pred = [event([1], [([2], 'x')])]
        result = score_documents([gold], [pred])
        self.assert_count(result, 'AC', 1, 0, 0)
        self.assert_count(result, 'EC_I', 0, 0, 1)
        self.assert_count(result, 'EC_C', 0, 0, 1)

    def test_official_agreement_without_collisions(self):
        spec = importlib.util.spec_from_file_location('original_utils', ROOT / 'utils.py')
        official = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(official)
        gold = [event([1], [([2], 'x'), ([5], 'y')]), event([2], [([8], 'z')]),
                event([3], [([7], 'x')], 'B')]
        pred = [event([1], [([2], 'wrong'), ([6], 'y')]), event([2], [([8], 'z')]),
                event([3], [([7], 'x')], 'C'), event([9], [([1], 'y')])]
        for mode in ('offset', 'token'):
            scores = score_documents([gold], [pred], mode)['metrics']
            for key, old in zip(METRIC_KEYS, official.calculate_F1([gold], [pred], mode)):
                self.assertAlmostEqual(scores[key], old, places=14)

    def test_full_gold_self_and_reversed_events(self):
        for split in ('dev', 'test'):
            docs = [d['events'] for d in json.loads((ROOT / 'data/SciEvents' / (split + '.json')).read_text())]
            pred = [[dict(e, arguments=list(reversed(e['arguments']))) for e in reversed(d)] for d in docs]
            for mode in ('offset', 'token'):
                result = score_documents(docs, pred, mode)
                self.assertTrue(all(x == 1 for x in result['metrics'].values()), (split, mode))

    def test_invalid_mode_length_and_missing_expression(self):
        with self.assertRaises(ValueError):
            score_documents([[]], [])
        with self.assertRaises(ValueError):
            score_documents([], [], 'bogus')
        with self.assertRaises(ValueError):
            score_documents([[{'event_type': 'A', 'trigger': {}, 'arguments': []}]], [[]])
        result = score_documents([[]], [[]])
        self.assertTrue(all(v == 0 for v in result['metrics'].values()))


if __name__ == '__main__':
    unittest.main()
