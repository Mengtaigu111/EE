import json
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))

from align_events import align_forest
from build_sentence_dataset import build_rows_for_document
from linearize import parse_forest
from merge_sentence_predictions import merge_sentence_rows
from run_c_queue import generation_budget
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
import utils


class SentenceMergeTests(unittest.TestCase):
    def test_generation_budget_uses_sentence_audit(self):
        self.assertEqual(generation_budget({'max_assistant_tokens': 100}), 141)
        self.assertEqual(generation_budget({'max_assistant_tokens': 100}, explicit=512), 512)

    def test_trigger_scope_rejects_same_text_in_other_sentence(self):
        tokens = 'we test . they test .'.split()
        sentences = ['we test .', 'they test .']
        events = [{'type': 'ITT', 'trigger': 'test', 'args': []}]
        aligned, _ = align_forest(tokens, sentences, events, trigger_scopes=[(3, 6)])
        self.assertEqual(aligned[0]['trigger']['offsets'], [4])

    def test_repeated_same_type_trigger_uses_next_occurrence_for_new_arguments(self):
        tokens = 'we test and test .'.split()
        events = [{'type': 'FAC', 'trigger': 'test', 'args': [{'role': 'Subject', 'text': 'we'}]},
                  {'type': 'FAC', 'trigger': 'test', 'args': [{'role': 'Subject', 'text': 'and'}]}]
        aligned, _ = align_forest(tokens, [' '.join(tokens)], events,
                                   trigger_scopes=[(0, 5), (0, 5)])
        self.assertEqual([event['trigger']['offsets'] for event in aligned], [[1], [3]])

    def test_merge_aligns_cross_sentence_argument_and_deduplicates_exact_event(self):
        rows = [
            {'id': 'doc::sentence-0', 'document_id': 'doc', 'sentence_index': 0,
             'tokens': 'we build a model . it improves baseline .'.split(),
             'sentences': ['we build a model .', 'it improves baseline .'],
             'target_bounds': [0, 5]},
            {'id': 'doc::sentence-1', 'document_id': 'doc', 'sentence_index': 1,
             'tokens': 'we build a model . it improves baseline .'.split(),
             'sentences': ['we build a model .', 'it improves baseline .'],
             'target_bounds': [5, 9]},
        ]
        predictions = [
            {'id': 'doc::sentence-0', 'document_id': 'doc', 'sentence_index': 0,
             'parsed_events': [{'type': 'PRP', 'trigger': 'build',
                                'args': [{'role': 'Content', 'text': 'a model'}]}]},
            {'id': 'doc::sentence-1', 'document_id': 'doc', 'sentence_index': 1,
             'parsed_events': [{'type': 'FAC', 'trigger': 'improves',
                                'args': [{'role': 'Subject', 'text': 'a model'}]},
                               {'type': 'FAC', 'trigger': 'improves',
                                'args': [{'role': 'Subject', 'text': 'a model'}]}]},
        ]
        merged = merge_sentence_rows(rows, predictions)
        self.assertEqual(len(merged), 1)
        events = merged[0]['aligned_events']
        self.assertEqual(len(events), 2)
        self.assertEqual(events[1]['trigger']['offsets'], [6])
        self.assertEqual(events[1]['arguments'][0]['offsets'], [2, 3])
        self.assertEqual(merged[0]['merge_diagnostics']['duplicate_events'], 1)

    def test_gold_dev_sentence_roundtrip_uses_official_metric(self):
        project = Path(__file__).resolve().parents[3]
        docs = json.loads((project / 'data/SciEvents/dev.json').read_text(encoding='utf-8'))
        rows, predictions = [], []
        for doc in docs:
            doc_rows, _ = build_rows_for_document(doc, 'dev')
            rows.extend(doc_rows)
            for row in doc_rows:
                parsed, diagnostics = parse_forest(row['completion'])
                self.assertFalse(diagnostics['invalid'])
                predictions.append({'id': row['id'], 'parsed_events': parsed})
        merged = merge_sentence_rows(rows, predictions, deduplicate=False)
        gold_by_id = {doc['doc_id']: doc['events'] for doc in docs}
        golds = [gold_by_id[row['id']] for row in merged]
        preds = [row['aligned_events'] for row in merged]
        scores = utils.calculate_F1(golds, preds, 'offset')
        self.assertGreater(scores[0], 0.98)
        self.assertGreater(scores[3], 0.94)
        self.assertGreater(scores[5], 0.90)

    def test_semantic_gold_dev_sentence_roundtrip_uses_official_metric(self):
        project = Path(__file__).resolve().parents[3]
        docs = json.loads((project / 'data/SciEvents/dev.json').read_text(encoding='utf-8'))
        rows, predictions = [], []
        for doc in docs:
            doc_rows, _ = build_rows_for_document(doc, 'dev', semantic_path=True)
            rows.extend(doc_rows)
            for row in doc_rows:
                parsed, diagnostics = parse_forest(row['completion'])
                self.assertFalse(diagnostics['invalid'])
                self.assertEqual(diagnostics['path_inconsistent'], 0)
                predictions.append({'id': row['id'], 'parsed_events': parsed})
        merged = merge_sentence_rows(rows, predictions, deduplicate=False)
        gold_by_id = {doc['doc_id']: doc['events'] for doc in docs}
        golds = [gold_by_id[row['id']] for row in merged]
        preds = [row['aligned_events'] for row in merged]
        scores = utils.calculate_F1(golds, preds, 'offset')
        self.assertGreater(scores[0], 0.98)
        self.assertGreater(scores[3], 0.94)
        self.assertGreater(scores[5], 0.90)


if __name__ == '__main__':
    unittest.main()
