"""CPU failure gates for the independent sorted-first orchestration."""
import json
from pathlib import Path
import tempfile
import unittest
from run_sorted_first import select_checkpoint, validate_manifest, verify_locks
from verify_sorted_first_run import validate_observed_order, compare_scores, coverage, fixed_evaluation_modules

class RunnerTests(unittest.TestCase):
    def test_later_checkpoint_wins_tie(self):
        self.assertEqual(select_checkpoint([{'step':802,'AC_f1':.5},{'step':1604,'AC_f1':.5}])['step'],1604)
    def test_reject_nonfinite_selection(self):
        with self.assertRaises(ValueError): select_checkpoint([{'step':802,'AC_f1':float('nan')}])
    def test_budget_gate(self):
        m={'seed':42,'total_optimizer_steps':4010,'effective_batch_size':16,'steps_per_epoch':802,'checkpoint_steps':[802,1604,2406,3208,4010]}
        validate_manifest(m)
        m['total_optimizer_steps']=4000
        with self.assertRaises(ValueError): validate_manifest(m)
    def test_lock_detects_mutation(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'x';p.write_text('before')
            from run_sorted_first import sha
            locks={str(p):sha(p)};p.write_text('after')
            with self.assertRaises(ValueError): verify_locks(locks)
    def test_observed_order_gate(self):
        rows=[{'epoch':0,'index':i,'id':str(i)} for i in range(4)]
        rows += [{'epoch':1,'index':i,'id':str(i)} for i in [2,3,0,1]]
        expected=[list(range(4)),[2,3,0,1]]
        validate_observed_order(rows,[str(i) for i in range(4)],expected)
        rows[0]['id']='bad'
        with self.assertRaises(ValueError): validate_observed_order(rows,[str(i) for i in range(4)],expected)
    def test_missing_or_duplicate_prediction_ids(self):
        with self.assertRaises(ValueError): coverage([{'id':'a'},{'id':'a'}],[{'id':'a'},{'id':'b'}],'test')
    def test_fixed_import_ignores_stale_cache(self):
        import sys, types
        stale=types.ModuleType("evaluate");stale.__file__="/old/evaluate.py"
        old=sys.modules.get("evaluate");sys.modules["evaluate"]=stale
        try:
            evaluation,merge=fixed_evaluation_modules()
            self.assertIn("evaluation-v2-20261003/scripts",evaluation.__file__)
            self.assertIn("evaluation-v2-20261003/scripts",merge.__file__)
            self.assertIsNot(evaluation,stale)
        finally:
            if old is None: sys.modules.pop("evaluate",None)
            else: sys.modules["evaluate"]=old
    def test_metric_tampering_gate(self):
        with self.assertRaises(ValueError): compare_scores({'AC_f1':.5},{'AC_f1':.6},'test')

if __name__=='__main__': unittest.main()
