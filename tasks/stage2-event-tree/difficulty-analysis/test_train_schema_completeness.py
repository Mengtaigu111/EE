import unittest
from analyze_train_schema_completeness import event_completeness, sentence_summary

SCHEMA={'PRP':('Proposer','Content','Target')}

class CompletenessTests(unittest.TestCase):
    def event(self, roles):
        return {'event_type':'PRP','arguments':[{'argument_type':r} for r in roles]}
    def test_complete_prp(self):
        self.assertEqual(event_completeness(self.event(['Proposer','Content','Target']),SCHEMA)['score'],1.0)
    def test_partial_prp(self):
        self.assertAlmostEqual(event_completeness(self.event(['Proposer','Content']),SCHEMA)['score'],2/3)
    def test_repeated_role_counts_once(self):
        self.assertAlmostEqual(event_completeness(self.event(['Target','Target']),SCHEMA)['score'],1/3)
    def test_empty_root_sentence_is_na(self):
        self.assertEqual(sentence_summary({} if False else [])['min_schema_completeness'],None)
        self.assertEqual(sentence_summary([])['event_count'],0)
    def test_invalid_role_is_detectable(self):
        result=event_completeness(self.event(['Proposer','BadRole']),SCHEMA)
        self.assertEqual(result['invalid'],['BadRole'])
        self.assertAlmostEqual(result['score'],1/3)

if __name__=='__main__': unittest.main()
