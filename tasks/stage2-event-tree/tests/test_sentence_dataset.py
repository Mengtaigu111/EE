import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))

from build_sentence_dataset import build_rows_for_document
from prompts import make_appendix_sentence_prompt, make_semantic_sentence_prompt, make_sentence_prompt


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
             'trigger': {'text': 'outperforms', 'offsets': [7]},
             'arguments': [{'text': 'a model', 'offsets': [2, 3],
                            'argument_type': 'Arg1', 'nugget_type': 'APP'},
                           {'text': 'a baseline', 'offsets': [8, 9],
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

    def test_semantic_rows_use_semantic_prompt_and_completion(self):
        rows, _ = build_rows_for_document(example_doc(), 'dev', semantic_path=True)

        self.assertEqual(rows[0]['mode'], 'sentence-semantic-path')
        self.assertIn('Semantic Event Tree', rows[0]['prompt'])
        self.assertEqual(
            rows[0]['completion'],
            '(ROOT (Methodology (ProposerIntroducesContent (PRP (Trigger "propose") (Content "a model")))))',
        )
        self.assertEqual(
            rows[1]['completion'],
            '(ROOT (Results (ExplicitComparison (CMP (Trigger "outperforms") '
            '(Arg1 "a model") (Arg2 "a baseline")))))',
        )
        self.assertEqual(rows[2]['completion'], '(ROOT)')

    def test_semantic_sentence_prompt_is_separate_from_baseline_prompt(self):
        doc = example_doc()
        abstract = ' '.join(doc['document'])
        semantic = make_semantic_sentence_prompt(abstract, doc['sentences'], 1)
        baseline = make_sentence_prompt(abstract, doc['sentences'], 1)

        self.assertIn('Semantic Event Tree', semantic)
        self.assertIn('Semantic Prototypes', semantic)
        self.assertIn('Methodology\n  -> ProposerIntroducesContent -> PRP', semantic)
        self.assertIn('Do not output YES/NO decisions.', semantic)
        self.assertIn('TARGET SENTENCE [S2]:\nthe model outperforms a baseline .', semantic)
        self.assertNotIn('Event schema (event type: roles):', semantic)
        self.assertIn('Event schema (event type: roles):', baseline)
        self.assertNotIn('Semantic Event Tree', baseline)

    def test_appendix_schema_prompt_uses_exact_f2_reference_not_ontology_gloss(self):
        doc = example_doc()
        abstract = ' '.join(doc['document'])
        prompt = make_appendix_sentence_prompt(abstract, doc['sentences'], 1)

        self.assertIn('Appendix F.2 exact event schema', prompt)
        self.assertIn('ITT (IntroduceTarget)', prompt)
        self.assertIn(
            'Template: <Target:arg1> is the abstract research target under '
            '<Condition:arg2> circumstance on <Dataset:arg3> datasets in this paper.',
            prompt,
        )
        self.assertIn('WKS (WorkStatement)', prompt)
        self.assertIn('FIN (Finding)', prompt)
        self.assertIn('find or demostrate findings', prompt)
        self.assertNotIn('finding statement introducing a fact or comparison', prompt)
        self.assertNotIn('investigation or analysis of a target', prompt)
        self.assertIn('TARGET SENTENCE [S2]:\nthe model outperforms a baseline .', prompt)

    def test_appendix_schema_rows_keep_c_tree_target_but_use_appendix_prompt(self):
        rows, _ = build_rows_for_document(example_doc(), 'dev', appendix_schema=True)

        self.assertEqual(rows[0]['mode'], 'sentence-appendix-schema')
        self.assertIn('Appendix F.2 exact event schema', rows[0]['prompt'])
        self.assertEqual(rows[0]['completion'], '(ROOT (PRP (Trigger "propose") (Content "a model")))')
        self.assertEqual(
            rows[1]['completion'],
            '(ROOT (CMP (Trigger "outperforms") (Arg1 "a model") (Arg2 "a baseline")))',
        )
        self.assertEqual(rows[2]['completion'], '(ROOT)')


if __name__ == '__main__':
    unittest.main()
