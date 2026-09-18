import sys
import unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from evaluate import evaluate_rows


def row(gold, pred, invalid=False):
    return {'id':'d:0','document_id':'d','tokens':['our','good','model'], 'document_start':10,
            'gold_mentions':gold, 'parsed_mentions':pred, 'invalid':invalid}

class MetricTests(unittest.TestCase):
    def test_perfect_multitype_span_untyped_dedup(self):
        gold=[{'text':'model','type':t,'offsets':[12]} for t in ['APP','MOD']]
        m, _ = evaluate_rows([row(gold,[{'text':'model','type':t} for t in ['APP','MOD']])])
        self.assertEqual(m['typed']['tp'],2)
        self.assertEqual(m['untyped']['tp'],1)
        self.assertEqual(m['typed']['f1'],1)

    def test_unaligned_is_false_positive(self):
        m, _ = evaluate_rows([row([], [{'text':'missing','type':'APP'}])])
        self.assertEqual(m['typed']['fp'],1)
        self.assertEqual(m['untyped']['fp'],1)
        self.assertEqual(m['unaligned_rate'],1)

    def test_type_only_error(self):
        m, _ = evaluate_rows([row([{'text':'model','type':'APP','offsets':[12]}],[{'text':'model','type':'MOD'}])])
        self.assertEqual(m['untyped']['f1'],1)
        self.assertEqual(m['typed']['f1'],0)
        self.assertEqual(m['errors']['FN']['TYPE_ONLY'],1)
        self.assertEqual(m['errors']['FP']['OTHER'],1)

    def test_boundary_and_missing(self):
        gold=[{'type':'APP','offsets':[11,12]}, {'type':'OG','offsets':[10]}]
        m, _ = evaluate_rows([row(gold,[{'text':'model','type':'APP'}])])
        self.assertEqual(m['errors']['FN'],{'BOUNDARY':1,'TYPE_ONLY':0,'MISSING':1})
        self.assertEqual(m['errors']['FP']['BOUNDARY'],1)

    def test_invalid_empty(self):
        m, _ = evaluate_rows([row([{'type':'APP','offsets':[12]}],[],True)])
        self.assertEqual(m['invalid_rate'],1)
        self.assertEqual(m['typed']['fn'],1)

if __name__ == '__main__': unittest.main()
