import unittest
from build_sorted_first_data import order_rows

class OrderingTests(unittest.TestCase):
    def test_positive_monotonic_root_spread_and_exact_coverage(self):
        rows=[{'id':str(i),'n_events':int(i%3!=0)} for i in range(30)]
        scores={r['id']:(i%4)/4 if r['n_events'] else None for i,r in enumerate(rows)}
        ordered=order_rows(rows,scores,42)
        self.assertEqual({r['id'] for r in ordered},{r['id'] for r in rows})
        vals=[scores[r['id']] for r in ordered if r['n_events']]
        self.assertEqual(vals,sorted(vals))
        roots=[i for i,r in enumerate(ordered) if not r['n_events']]
        self.assertTrue(all(2<=b-a<=4 for a,b in zip(roots,roots[1:])))
        self.assertEqual(ordered,order_rows(rows,scores,42))
        self.assertNotEqual(ordered,order_rows(rows,scores,43))

    def test_missing_and_duplicate_ids_fail(self):
        with self.assertRaises(ValueError):order_rows([{'id':'a','n_events':1}],{},42)
        with self.assertRaises(ValueError):order_rows([{'id':'a','n_events':1}]*2,{'a':.1},42)

if __name__=='__main__':unittest.main()
