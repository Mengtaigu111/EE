import json
import sys
import unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from evaluate import evaluate
from linearize import canonical_events

PROJECT = Path(__file__).resolve().parents[3]


class EvaluateTests(unittest.TestCase):
    def test_oracle_and_empty(self):
        docs = json.loads((PROJECT / 'data/SciEvents/dev.json').read_text())[:20]
        rows = [{'id': d['doc_id'], 'split': 'dev', 'tokens': d['document'], 'sentences': d['sentences'], 'gold_events': d['events']} for d in docs]
        metrics, out, per_doc = evaluate(rows, None, oracle=True)
        self.assertGreater(metrics['official']['AC_f1'], 0.95)
        self.assertEqual(len(out), 20)
        self.assertEqual(metrics['scope'], 'full')
        empty = [{'id': r['id'], 'parsed_events': [], 'invalid': True, 'parse_diagnostics': {'invalid': True}} for r in rows]
        metrics, _, _ = evaluate(rows, empty)
        self.assertEqual(metrics['official']['TI_f1'], 0.0)
        self.assertEqual(metrics['alignment']['invalid_documents'], 20)

    def test_partial_prediction(self):
        d = json.loads((PROJECT / 'data/SciEvents/dev.json').read_text())[0]
        rows = [{'id': d['doc_id'], 'split': 'dev', 'tokens': d['document'], 'sentences': d['sentences'], 'gold_events': d['events']}]
        events = canonical_events(d)[:1]
        pred = [{'id': d['doc_id'], 'parsed_events': [{'type': e['type'], 'trigger': e['trigger'], 'args': [{'role': a['role'], 'text': a['text']} for a in e['args']]} for e in events], 'invalid': False, 'parse_diagnostics': {}}]
        metrics, _, _ = evaluate(rows, pred)
        self.assertEqual(metrics['official']['TI_p'], 1.0)
        self.assertLess(metrics['official']['TI_r'], 1.0)


if __name__ == '__main__':
    unittest.main()
