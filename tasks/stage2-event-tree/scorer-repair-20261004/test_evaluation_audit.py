import unittest
from run_evaluation_audit import residual_text_matches, order_documents, assert_grounded_mentions, verify_hash_sets

def e(text,offsets):return {'event_type':'PRP','trigger':{'tokens':text.split(),'offsets':offsets},'arguments':[]}

class AuditTests(unittest.TestCase):
    def test_only_residual_same_text_different_position_counts(self):
        gold=[e('find',[1]),e('find',[8]),e('report',[12])]
        pred=[e('find',[1]),e('find',[6]),e('other',[12])]
        matched=residual_text_matches(gold,pred)
        self.assertEqual([(x['gold_index'],x['prediction_index']) for x in matched],[(1,1)])
        self.assertNotEqual(matched[0]['gold_offsets'],matched[0]['prediction_offsets'])

    def test_grounding_mismatch_is_rejected(self):
        with self.assertRaises(ValueError):
            assert_grounded_mentions([{'document':['find']}],[[e('report',[0])]])

    def test_historical_hash_cannot_be_overwritten(self):
        import tempfile
        from pathlib import Path
        from run_evaluation_audit import sha
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'x';p.write_text('changed')
            with self.assertRaises(ValueError):
                verify_hash_sets({str(p):'old expected'}, {str(p):sha(p)})

    def test_document_coverage_rejects_missing_and_duplicate(self):
        gold=[{'doc_id':'a','events':[]},{'doc_id':'b','events':[]}]
        with self.assertRaises(ValueError):order_documents(gold,[{'doc_id':'a','events':[]}])
        with self.assertRaises(ValueError):order_documents(gold,[{'doc_id':'a','events':[]}]*2)
        self.assertEqual(order_documents(gold,[{'doc_id':'b','events':[]},{'doc_id':'a','events':[]}]),[[],[]])

if __name__=='__main__':unittest.main()
