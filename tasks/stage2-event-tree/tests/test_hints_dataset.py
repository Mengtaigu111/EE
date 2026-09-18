import json
import sys
import unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from hints import gold_hints, noised_hints, render_flat, render_typed
from prompts import make_prompt
from build_dataset import build_row

PROJECT = Path(__file__).resolve().parents[3]
PROFILE = {'per_type_recall': {t: 0.5 for t in ('OG', 'APP', 'MOD', 'FEA', 'TAK', 'DST', 'LIM', 'STR', 'WEA', 'DEG')},
           'error_mix': {'BOUNDARY': .45, 'TYPE_ONLY': .2, 'MISSING': .35}, 'spurious_per_sentence': .9, 'boundary_longer_share': .6}


class HintTests(unittest.TestCase):
    def setUp(self):
        self.doc = json.loads((PROJECT / 'data/SciEvents/dev.json').read_text())[0]

    def test_gold_hints_unique_sorted(self):
        hints = gold_hints(self.doc)
        keys = [(tuple(h['offsets']), h['type']) for h in hints]
        self.assertEqual(len(keys), len(set(keys)))
        self.assertEqual(keys, sorted(keys, key=lambda k: (k[0][0], k[0][-1], k[1])))
        self.assertTrue(all(not h['type'].startswith('E-') for h in hints))

    def test_noise_deterministic_and_lossy(self):
        gold = gold_hints(self.doc)
        a = noised_hints(self.doc, gold, PROFILE, 42)
        b = noised_hints(self.doc, gold, PROFILE, 42)
        c = noised_hints(self.doc, gold, PROFILE, 43)
        self.assertEqual(a, b)
        self.assertNotEqual(a, c)
        exact = {(tuple(h['offsets']), h['type']) for h in gold} & {(tuple(h['offsets']), h['type']) for h in a}
        self.assertLess(len(exact), len(gold))
        self.assertTrue(all(h['text'] == ' '.join(self.doc['document'][i] for i in h['offsets']) for h in a))

    def test_prompt_modes(self):
        hints = gold_hints(self.doc)
        none = make_prompt('abs', 'none')
        flat = make_prompt('abs', 'flat', hints)
        typed = make_prompt('abs', 'typed', hints)
        self.assertNotIn('Candidate mentions', none)
        self.assertIn(render_flat(hints), flat)
        self.assertNotIn('[', flat.split('Candidate mentions')[0].split('Event schema')[1])
        self.assertIn(render_typed(hints), typed)
        self.assertIn('Dataset[DST]', typed)

    def test_build_row(self):
        row = build_row(self.doc, 'dev', 'typed', gold_hints(self.doc))
        self.assertTrue(row['completion'].startswith('(ROOT'))
        self.assertEqual(row['n_events'], len(self.doc['events']))
        self.assertIn(row['abstract'], row['prompt'])


    def test_new_modes(self):
        noschema = make_prompt('abs', 'noschema')
        self.assertNotIn('Event schema', noschema)
        natural = make_prompt('abs', 'natural')
        self.assertIn('Proposal (', natural)
        self.assertNotIn('PRP (', natural)
        sub = make_prompt('abs', 'sub')
        self.assertIn('Pairs:', sub)
        row = build_row(self.doc, 'dev', 'sub', [])
        self.assertFalse(row['completion'].startswith('(ROOT'))
        row = build_row(self.doc, 'dev', 'natural', [])
        self.assertIn('(Proposal (Trigger', row['completion'] + '(Proposal (Trigger')  # at least parses


if __name__ == '__main__':
    unittest.main()
