import unittest
from score_raw_token import convert_events, merge_predictions

class TokenInputTests(unittest.TestCase):
    def test_keep_unmatchable_and_duplicate_events(self):
        event={'type':'PRP','trigger':'not in input','args':[{'role':'Content','text':'modified Text'}]}
        result=convert_events([event,event])
        self.assertEqual(len(result),2)
        self.assertEqual(result[0]['trigger']['tokens'],['not','in','input'])
        self.assertEqual(result[0]['arguments'][0]['tokens'],['modified','Text'])
        self.assertNotIn('offsets',result[0]['trigger'])

    def test_sentence_order_and_missing_prediction_gate(self):
        rows=[{'id':'b','document_id':'d','sentence_index':1}, {'id':'a','document_id':'d','sentence_index':0}]
        preds=[{'id':'b','raw_output':'(ROOT (FIN (Trigger "b")))'}, {'id':'a','raw_output':'(ROOT (PRP (Trigger "a")))'}]
        result,_=merge_predictions(rows,preds)
        self.assertEqual([x['trigger']['tokens'] for x in result['d']],[['a'],['b']])
        with self.assertRaises(ValueError):
            merge_predictions(rows,preds[:1])

if __name__=='__main__': unittest.main()
