import unittest

from analyze_difficulty import (
    argument_geometry,
    match_events,
    multiset_scores,
    schema_similarity,
)


class DifficultyAnalysisTests(unittest.TestCase):
    def test_discontinuous_offsets_report_runs_and_internal_gap(self):
        result = argument_geometry([2, 3, 6, 10, 11])
        self.assertEqual(result['fragment_count'], 3)
        self.assertEqual(result['gap_tokens'], 5)
        self.assertEqual(result['max_gap_tokens'], 3)

    def test_nested_is_strict_containment_not_exact_reuse(self):
        self.assertTrue(argument_geometry([2, 3, 4])['contiguous'])
        self.assertFalse(argument_geometry([2, 3, 4])['discontinuous'])

    def test_schema_similarity_uses_role_sets(self):
        self.assertAlmostEqual(schema_similarity({'Aim', 'Condition'}, {'Aim', 'Condition', 'Dataset'}, {'Aim': 1, 'Condition': 1, 'Dataset': 1}), 2 / 3)

    def test_event_matching_pairs_same_trigger_even_when_type_is_wrong(self):
        gold = [{'event_type': 'RWS', 'trigger': {'text': 'use', 'offsets': [4]}, 'arguments': []}]
        pred = [{'event_type': 'MDS', 'trigger': {'text': 'use', 'offsets': [4]}, 'arguments': []}]
        matches, missed, spurious = match_events(gold, pred)
        self.assertEqual([(g, p) for g, p, _ in matches], [(0, 0)])
        self.assertEqual(missed, [])
        self.assertEqual(spurious, [])

    def test_type_multiset_scores_preserve_duplicate_types(self):
        result = multiset_scores(['PUR', 'PUR', 'MDS'], ['PUR', 'MDS', 'WKS'])
        self.assertEqual((result['tp'], result['fp'], result['fn']), (2, 1, 1))
        self.assertEqual(result['exact'], 0)


if __name__ == '__main__':
    unittest.main()
