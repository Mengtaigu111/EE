import copy
import importlib.util
import json
from pathlib import Path
import sys
import unittest

SCRIPTS = Path(__file__).resolve().parents[1] / 'scripts'
sys.path.insert(0, str(SCRIPTS))


def fixture():
    tokens = 'we improve a model . empty .'.split()
    def arg(offsets, kind):
        return {'offsets': offsets, 'nugget_type': kind,
                'text': ' '.join(tokens[i] for i in offsets)}
    raw = {'doc_id': 'd', 'document': tokens,
           'sentences': ['we improve a model .', 'empty .'],
           'events': [{'trigger': {'offsets': [1]}, 'arguments': [
               arg([1], 'STR'), arg([0], 'OG'), arg([1], 'E-FAC'),
               arg([2, 3], 'APP'), arg([2, 3], 'APP'), arg([2, 3], 'MOD'),
               arg([0, 5], 'FEA'), arg([3, 0, 3], 'TAK')]}]}
    normalized = {'document_id': 'd', 'tokens': tokens, 'text': ' '.join(tokens),
                  'sentences': [{'sentence_id': 'd:0', 'sentence_index': 0,
                                 'start': 0, 'end': 5, 'text': raw['sentences'][0]},
                                {'sentence_id': 'd:1', 'sentence_index': 1,
                                 'start': 5, 'end': 7, 'text': raw['sentences'][1]}]}
    return raw, normalized


class DatasetRules(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        assert importlib.util.find_spec('build_dataset'), 'build_dataset implementation missing'
        import build_dataset
        cls.module = build_dataset

    def test_typed_gold_scope_and_order(self):
        raw, normalized = fixture()
        rows, stats, audit = self.module.build_document(raw, normalized, 'train', 'S_G')
        self.assertEqual([(x['text'], x['type']) for x in rows[0]['gold_mentions']],
                         [('we', 'OG'), ('we model', 'TAK'), ('improve', 'STR'),
                          ('a model', 'APP'), ('a model', 'MOD')])
        self.assertEqual(rows[1]['completion'], '{"mentions":[]}')
        self.assertEqual(stats['duplicate_typed_annotations'], 1)
        self.assertEqual(stats['cross_sentence'], 1)
        self.assertEqual(stats['retained_sentence_local'], 5)
        self.assertEqual(stats['local_discontinuous'], 1)
        self.assertEqual({x['reason'] for x in audit}, {'cross_sentence', 'canonicalized'})
        self.assertEqual(rows[0]['gold_mentions'][0]['annotation_order'], 2)
        self.assertEqual(rows[1]['document_start'], 5)

    def test_sentence_boundary_corruption_is_rejected(self):
        raw, norm = fixture()
        norm['sentences'][0]['end'] = 4
        with self.assertRaises(ValueError):
            self.module.build_document(raw, norm, 'train', 'S_G')

    def test_modes_have_identical_gold_and_context_only_in_sd(self):
        raw, norm = fixture()
        s, _, _ = self.module.build_document(raw, norm, 'train', 'S_G')
        sd, _, _ = self.module.build_document(raw, norm, 'train', 'SD_G')
        self.assertEqual(s[0]['completion'], sd[0]['completion'])
        self.assertNotIn('Abstract:', s[0]['prompt'])
        self.assertIn('Abstract:\n' + norm['text'], sd[0]['prompt'])
        self.assertEqual(set(json.loads(sd[0]['completion'])['mentions'][0]), {'text', 'type'})

    def test_four_examples_are_deterministic_and_cover_all_types(self):
        rows = [{'id': str(i), 'gold_mentions': [{'type': t} for t in types]}
                for i, types in enumerate([['APP', 'TAK', 'OG'], ['MOD', 'FEA', 'DST'],
                                          ['LIM', 'STR', 'WEA'], ['DEG'], ['APP']])]
        chosen = self.module.select_examples(rows)
        self.assertEqual(len(chosen), 4)
        self.assertEqual(len({m['type'] for r in chosen for m in r['gold_mentions']}), 10)
        self.assertEqual(chosen, self.module.select_examples(rows))

    def test_fewshot_contains_only_example_sentence_output(self):
        from prompts import make_prompt
        prompt = make_prompt('target', 'context', [{'target_sentence': 'example',
                             'completion': '{"mentions":[]}', 'abstract': 'SECRET'}])
        self.assertIn('Example 1\nTarget sentence:\nexample\n\nOutput:\n{"mentions":[]}', prompt)
        self.assertNotIn('SECRET', prompt)
        self.assertTrue(prompt.endswith('Target sentence:\ntarget'))


if __name__ == '__main__':
    unittest.main()
