import json
import sys
import unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from linearize import canonical_events, linearize, linearize_substructures, parse_forest

PROJECT = Path(__file__).resolve().parents[3]


def doc(tokens, events):
    return {'doc_id': 'd', 'document': tokens.split(), 'sentences': [tokens], 'events': events}


class LinearizeTests(unittest.TestCase):
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
                                                    'unknown_event_types': 0, 'unknown_roles': 0, 'missing_trigger': 0}))

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


if __name__ == '__main__':
    unittest.main()
