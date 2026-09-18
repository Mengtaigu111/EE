import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))

from build_sentence_dataset import build_rows_for_document


def example_doc():
    sentences = [
        'we propose a model .',
        'the model outperforms a baseline .',
        'this sentence has no event .',
    ]
    tokens = ' '.join(sentences).split()
    return {
        'doc_id': 'doc-c',
        'document': tokens,
        'sentences': sentences,
        'events': [
            {'event_type': 'PRP',
             'trigger': {'text': 'propose', 'offsets': [1]},
             'arguments': [{'text': 'a model', 'offsets': [2, 3],
                            'argument_type': 'Content', 'nugget_type': 'APP'}]},
            {'event_type': 'CMP',
             'trigger': {'text': 'outperforms', 'offsets': [8]},
             'arguments': [{'text': 'a model', 'offsets': [2, 3],
                            'argument_type': 'Arg1', 'nugget_type': 'APP'},
                           {'text': 'a baseline', 'offsets': [9, 10],
                            'argument_type': 'Arg2', 'nugget_type': 'APP'}]},
        ],
    }


class SentenceDatasetTests(unittest.TestCase):
    def test_rows_partition_events_and_keep_full_context(self):
        rows, stats = build_rows_for_document(example_doc(), 'dev')
        self.assertEqual([row['sentence_index'] for row in rows], [0, 1, 2])
        self.assertEqual([row['id'] for row in rows], ['doc-c::sentence-0', 'doc-c::sentence-1', 'doc-c::sentence-2'])
        self.assertEqual([row['n_events'] for row in rows], [1, 1, 0])
        self.assertEqual(rows[0]['target_bounds'], [0, 5])
        self.assertEqual(rows[1]['target_bounds'], [5, 11])
        self.assertIn('[S1]', rows[0]['prompt'])
        self.assertIn('[S2]', rows[0]['prompt'])
        self.assertIn('TARGET SENTENCE [S1]', rows[0]['prompt'])
        self.assertEqual(rows[1]['target_events'][0]['arguments'][0]['offsets'], [2, 3])
        self.assertEqual(rows[2]['completion'], '(ROOT)')
        self.assertEqual(stats['sentences'], 3)
        self.assertEqual(stats['no_event_sentences'], 1)
        self.assertEqual(stats['cross_sentence_arguments'], 1)
        self.assertEqual(stats['cross_sentence_events'], 1)


if __name__ == '__main__':
    unittest.main()
