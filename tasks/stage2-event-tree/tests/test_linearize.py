import json
import sys
import unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from linearize import canonical_events, linearize, linearize_semantic, linearize_substructures, parse_forest, span_text

PROJECT = Path(__file__).resolve().parents[3]


def doc(tokens, events):
    return {'doc_id': 'd', 'document': tokens.split(), 'sentences': [tokens], 'events': events}


class LinearizeTests(unittest.TestCase):
    def test_span_text_preserves_order_and_repeated_offsets(self):
        self.assertEqual(span_text(['knowledge', 'graph', 'two', 'datasets'], [2, 0, 1, 3]),
                         'two knowledge graph datasets')
        self.assertEqual(span_text(['very', 'good'], [0, 0, 1]), 'very very good')

    def test_arguments_preserve_gold_order_but_list_uses_source_position(self):
        d = doc('knowledge graph propose two datasets', [{
            'event_type': 'PRP', 'trigger': {'offsets': [2]},
            'arguments': [
                {'offsets': [1], 'argument_type': 'Content'},
                {'offsets': [3, 0, 1, 4], 'tokens': ['two', 'knowledge', 'graph', 'datasets'],
                 'text': 'two knowledge graph datasets', 'argument_type': 'Content'},
                {'offsets': [3, 3], 'tokens': ['two', 'two'],
                 'text': 'two two', 'argument_type': 'Content'}]}])
        before = json.dumps(d)
        args = canonical_events(d)[0]['args']
        self.assertEqual([a['text'] for a in args],
                         ['two knowledge graph datasets', 'graph', 'two two'])
        self.assertEqual(args[0]['offsets'], [3, 0, 1, 4])
        self.assertEqual(args[2]['offsets'], [3, 3])
        self.assertEqual(json.dumps(d), before)

    def test_inconsistent_gold_text_or_tokens_is_rejected(self):
        for field, value in [('text', 'graph two'), ('tokens', ['graph', 'two'])]:
            with self.subTest(field=field):
                d = doc('graph propose two', [{'event_type': 'PRP', 'trigger': {'offsets': [1]},
                    'arguments': [{'offsets': [2, 0], 'argument_type': 'Content', field: value}]}])
                with self.assertRaisesRegex(ValueError, 'gold.*'+field):
                    canonical_events(d)

    def test_real_nonmonotonic_gold_argument(self):
        docs = json.loads((PROJECT / 'data/SciEvents/test.json').read_text())
        d = next(d for d in docs if d['doc_id'] == 'f759023d55386ba9f605e8bc41778619')
        tree = linearize(canonical_events(d))
        self.assertIn('(Content "two knowledge graph datasets")', tree)
        self.assertNotIn('(Content "knowledge graph two datasets")', tree)

    def test_canonical_order_and_roundtrip(self):
        d = doc('we propose x and it outperforms y on z',
                [{'event_type': 'CMP', 'trigger': {'offsets': [5]},
                  'arguments': [{'offsets': [8], 'argument_type': 'Dataset', 'nugget_type': 'DST'},
                                {'offsets': [2], 'argument_type': 'Arg1', 'nugget_type': 'APP'}]},
                 {'event_type': 'PRP', 'trigger': {'offsets': [1]},
                  'arguments': [{'offsets': [2], 'argument_type': 'Content', 'nugget_type': 'APP'},
                                {'offsets': [0], 'argument_type': 'Proposer', 'nugget_type': 'OG'}]}])
        tree = linearize(canonical_events(d))
        self.assertEqual(tree, '(ROOT (PRP (Trigger "propose") (Proposer "we") (Content "x")) '
                               '(CMP (Trigger "outperforms") (Arg1 "x") (Dataset "z")))')
        events, diag = parse_forest(tree)
        self.assertFalse(diag['invalid'])
        self.assertEqual([e['type'] for e in events], ['PRP', 'CMP'])
        self.assertEqual(events[1]['args'], [{'role': 'Arg1', 'text': 'x'}, {'role': 'Dataset', 'text': 'z'}])

    def test_escaping(self):
        d = doc('a " b \\ c', [{'event_type': 'FAC', 'trigger': {'offsets': [1]},
                              'arguments': [{'offsets': [3], 'argument_type': 'Subject', 'nugget_type': 'APP'}]}])
        tree = linearize(canonical_events(d))
        events, diag = parse_forest(tree)
        self.assertEqual(events[0]['trigger'], '"')
        self.assertEqual(events[0]['args'][0]['text'], '\\')

    def test_tolerant_parser(self):
        events, diag = parse_forest('(ROOT (PRP (Trigger "propose") (Proposer "we") (Bogus "x")) (XYZ (Trigger "q")) (CMP (Arg1 "a"))')
        self.assertEqual(len(events), 1)
        self.assertEqual(diag['unknown_roles'], 1)
        self.assertEqual(diag['unknown_event_types'], 1)
        self.assertEqual(diag['missing_trigger'], 1)
        self.assertTrue(parse_forest('garbage')[1]['invalid'])
        self.assertEqual(parse_forest('(ROOT)'), ([], {'invalid': False, 'dropped_events': 0, 'dropped_args': 0,
                                                    'unknown_event_types': 0, 'unknown_roles': 0, 'missing_trigger': 0,
                                                    'semantic_events': 0, 'path_inconsistent': 0}))

    def test_full_dev_roundtrip(self):
        docs = json.loads((PROJECT / 'data/SciEvents/dev.json').read_text())
        for d in docs:
            events = canonical_events(d)
            parsed, diag = parse_forest(linearize(events))
            self.assertFalse(diag['invalid'] or diag['dropped_events'] or diag['dropped_args'], d['doc_id'])
            self.assertEqual(len(parsed), len(events))
            self.assertEqual(sum(len(e['args']) for e in parsed), sum(len(e['args']) for e in events))


    def test_natural_labels_and_substructures(self):
        d = doc('we propose x', [{'event_type': 'PRP', 'trigger': {'offsets': [1]},
                                 'arguments': [{'offsets': [0], 'argument_type': 'Proposer', 'nugget_type': 'OG'}]}])
        ev = canonical_events(d)
        tree = linearize(ev, natural=True)
        self.assertEqual(tree, '(ROOT (Proposal (Trigger "propose") (Proposer "we")))')
        events, diag = parse_forest(tree)
        self.assertEqual(events[0]['type'], 'PRP')
        self.assertEqual(linearize_substructures(ev), '(PRP "propose") (Proposer "we")')
        self.assertEqual(linearize_substructures([]), '()')

    def test_semantic_tree_roundtrip_and_path_diagnostics(self):
        d = doc('we propose x and it outperforms y on z',
                [{'event_type': 'PRP', 'trigger': {'offsets': [1]},
                  'arguments': [{'offsets': [0], 'argument_type': 'Proposer', 'nugget_type': 'OG'},
                                {'offsets': [2], 'argument_type': 'Content', 'nugget_type': 'APP'}]},
                 {'event_type': 'CMP', 'trigger': {'offsets': [5]},
                  'arguments': [{'offsets': [2], 'argument_type': 'Arg1', 'nugget_type': 'APP'},
                                {'offsets': [8], 'argument_type': 'Dataset', 'nugget_type': 'DST'}]}])
        tree = linearize_semantic(canonical_events(d))
        self.assertEqual(
            tree,
            '(ROOT (Methodology (ProposerIntroducesContent (PRP (Trigger "propose") '
            '(Proposer "we") (Content "x")))) (Results (ExplicitComparison '
            '(CMP (Trigger "outperforms") (Arg1 "x") (Dataset "z")))))')
        events, diag = parse_forest(tree)
        self.assertFalse(diag['invalid'])
        self.assertEqual(diag['semantic_events'], 2)
        self.assertEqual(diag['path_inconsistent'], 0)
        self.assertEqual([event['type'] for event in events], ['PRP', 'CMP'])
        self.assertEqual(events[0]['args'], [{'role': 'Proposer', 'text': 'we'}, {'role': 'Content', 'text': 'x'}])

    def test_semantic_parser_recovers_wrong_path_without_retyping_event(self):
        raw = '(ROOT (Results (SubjectOutcome (CMP (Trigger "beats") (Arg1 "a") (Arg2 "b")))))'
        events, diag = parse_forest(raw)

        self.assertEqual(events[0]['type'], 'CMP')
        self.assertEqual(diag['semantic_events'], 1)
        self.assertEqual(diag['path_inconsistent'], 1)


if __name__ == '__main__':
    unittest.main()
