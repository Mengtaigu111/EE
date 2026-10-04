"""Lossless positioned forest protocol regression tests; no model output repair."""
import copy
import json
from pathlib import Path
import unittest

from instance_scorer import score_documents
from lossless_forest import decode_forest, encode_forest

ROOT = Path(__file__).resolve().parents[3]


def mention(tokens, offsets):
    selected = [tokens[i] for i in offsets]
    return {'offsets': offsets, 'tokens': selected, 'text': ' '.join(selected)}


def event(tokens, offsets, args=(), kind='PRP'):
    return {'event_type': kind, 'trigger': mention(tokens, offsets),
            'arguments': [dict(mention(tokens, span), argument_type=role) for span, role in args]}


class LosslessForestTests(unittest.TestCase):
    def test_reverse_discontinuous_duplicate_indices_and_roles_roundtrip(self):
        tokens = ['we', 'propose', 'a', 'model', 'again']
        events = [event(tokens, [1], [([4, 2, 4], 'Content'), ([3, 0], 'Content')])]
        original = copy.deepcopy(events)
        raw = encode_forest(tokens, events)
        self.assertIn('(At 4 2 4)', raw)
        self.assertEqual(decode_forest(tokens, raw), original)
        self.assertEqual(events, original)

    def test_same_text_different_positions_and_duplicate_events_survive(self):
        tokens = ['propose', 'model', 'propose', 'model']
        events = [event(tokens, [0], [([1], 'Content')]),
                  event(tokens, [2], [([3], 'Content')]),
                  event(tokens, [0], [([1], 'Content')]),
                  event(tokens, [0], [([3], 'Content')], 'WKS')]
        self.assertEqual(decode_forest(tokens, encode_forest(tokens, events)), events)

    def test_quotes_backslashes_parentheses_unicode_and_json_escapes(self):
        tokens = ['say"this', r'back\slash', '(model)', '模型']
        events = [event(tokens, [0], [([1, 2, 3], 'Content')])]
        self.assertEqual(decode_forest(tokens, encode_forest(tokens, events)), events)
        raw = '(ROOT (PRP (Trigger "\\u6a21\\u578b" (At 3))))'
        self.assertEqual(decode_forest(tokens, raw)[0]['trigger']['tokens'], ['模型'])

    def test_manual_protocol_uses_ordered_global_offsets(self):
        tokens = ['a', 'same', 'same', 'b']
        raw = '(ROOT (PRP (Trigger "same" (At 2)) (Content "b a b" (At 3 0 3))))'
        self.assertEqual(decode_forest(tokens, raw),
                         [event(tokens, [2], [([3, 0, 3], 'Content')])])

    def test_empty_forest(self):
        self.assertEqual(encode_forest([], []), '(ROOT)')
        self.assertEqual(decode_forest([], '(ROOT)'), [])

    def test_missing_position_never_guesses(self):
        for raw in ['(ROOT (PRP (Trigger "a")))',
                    '(ROOT (PRP (Trigger "a" (At 0)) (Content "a")))']:
            with self.subTest(raw=raw), self.assertRaises(ValueError):
                decode_forest(['a'], raw)

    def test_rejects_bounds_text_mismatch_empty_offsets_and_invalid_index(self):
        for fragment in ['"a" (At 1)', '"a" (At -1)', '"wrong" (At 0)',
                         '"" (At 0)', '"a" (At)', '"a" (At 0.0)',
                         '"a" (At true)', '"a" (At 00)']:
            with self.subTest(fragment=fragment), self.assertRaises(ValueError):
                decode_forest(['a'], '(ROOT (PRP (Trigger ' + fragment + ')))')

    def test_rejects_schema_violations_missing_duplicate_and_reordered_trigger(self):
        raws = ['(ROOT (UNKNOWN (Trigger "a" (At 0))))',
                '(ROOT (PRP (Trigger "a" (At 0)) (Unknown "a" (At 0))))',
                '(ROOT (PRP (Trigger "a" (At 0)) (Finder "a" (At 0))))',
                '(ROOT (PRP (Trigger "a" (At 0)) ("" "a" (At 0))))',
                '(ROOT (PRP))',
                '(ROOT (PRP (Trigger "a" (At 0)) (Trigger "a" (At 0))))',
                '(ROOT (PRP (Content "a" (At 0)) (Trigger "a" (At 0))))']
        for raw in raws:
            with self.subTest(raw=raw), self.assertRaises(ValueError):
                decode_forest(['a'], raw)

    def test_strict_grammar_rejects_truncated_extra_trees_metadata_and_invalid_json(self):
        valid = '(ROOT (PRP (Trigger "a" (At 0))))'
        for raw in ['', valid[:-1], valid + ')', valid + '(ROOT)',
                    valid + ' junk', valid.replace('(At 0)', '(At 0) (At 0)'),
                    valid.replace('"a"', '"\\q"'), valid.replace('ROOT', 'OTHER'),
                    valid.replace('"a"', 'a')]:
            with self.subTest(raw=raw), self.assertRaises(ValueError):
                decode_forest(['a'], raw)

    def test_encode_checks_original_mentions_and_schema(self):
        original = event(['a'], [0])
        for change in ['missing', 'tokens', 'text', 'empty', 'out_of_bounds', 'bool']:
            e = copy.deepcopy(original)
            if change == 'missing':
                del e['trigger']['offsets']
            elif change == 'tokens':
                e['trigger']['tokens'] = ['wrong']
            elif change == 'text':
                e['trigger']['text'] = 'wrong'
            else:
                e['trigger']['offsets'] = {'empty': [], 'out_of_bounds': [1], 'bool': [True]}[change]
            with self.subTest(change=change), self.assertRaises(ValueError):
                encode_forest(['a'], [e])
        for e in [dict(original, event_type='UNKNOWN'),
                  event(['a'], [0], [([0], 'Finder')])]:
            with self.assertRaises(ValueError):
                encode_forest(['a'], [e])

    def test_full_original_gold_roundtrip_offsets_tokens_roles_and_metrics(self):
        for split in ('dev', 'test'):
            docs = json.loads((ROOT / 'data/SciEvents' / (split + '.json')).read_text())
            decoded = []
            for doc in docs:
                events = decode_forest(doc['document'], encode_forest(doc['document'], doc['events']))
                decoded.append(events)
                self.assertEqual(len(events), len(doc['events']))
                for before, after in zip(doc['events'], events):
                    self.assertEqual(before['event_type'], after['event_type'])
                    for key in ('offsets', 'tokens'):
                        self.assertEqual(before['trigger'][key], after['trigger'][key])
                    self.assertEqual(len(before['arguments']), len(after['arguments']))
                    for arg_before, arg_after in zip(before['arguments'], after['arguments']):
                        for key in ('offsets', 'tokens', 'argument_type'):
                            self.assertEqual(arg_before[key], arg_after[key])
            golds = [doc['events'] for doc in docs]
            for mode in ('offset', 'token'):
                scores = score_documents(golds, decoded, mode)
                self.assertTrue(all(v == 1 for v in scores['metrics'].values()), (split, mode))


if __name__ == '__main__':
    unittest.main()
