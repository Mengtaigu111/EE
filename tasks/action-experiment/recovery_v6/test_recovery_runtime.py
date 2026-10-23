import tempfile
from pathlib import Path
import unittest
from recovery_runtime import log_offsets,truncate_logs,resume_decision,reconcile_artifacts

class RecoveryRuntimeTests(unittest.TestCase):
    def test_early_stop_never_restarts_training(self):
        self.assertEqual(resume_decision(1600,6400,12828,1600,True,400),dict(evaluate=False,train=False))
    def test_final_dev_repeated_after_interruption(self):
        self.assertEqual(resume_decision(3207,12828,12828,3200,False,400),dict(evaluate=True,train=False))
    def test_unfinished_initial_and_regular_dev(self):
        self.assertTrue(resume_decision(0,0,8,-1,False,400)['evaluate'])
        self.assertTrue(resume_decision(1200,4800,12828,800,False,400)['evaluate'])
        self.assertFalse(resume_decision(1200,4800,12828,1200,False,400)['evaluate'])
    def test_partial_logs_are_archived_and_truncated(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp);f=p/'groups.jsonl';f.write_bytes(b'complete\n')
            offset=log_offsets(p)
            with f.open('ab') as out:out.write(b'partial\n')
            (p/'rollouts.jsonl').write_bytes(b'uncommitted\n')
            truncate_logs(p,offset)
            self.assertEqual(f.read_bytes(),b'complete\n')
            self.assertEqual((p/'rollouts.jsonl').read_bytes(),b'')
            tails=sorted(q.read_bytes() for q in (p/'interrupted-logs').rglob('*.jsonl'))
            self.assertEqual(tails,[b'partial\n',b'uncommitted\n'])
    def test_missing_log_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp);(p/'groups.jsonl').write_bytes(b'a')
            with self.assertRaises(ValueError):truncate_logs(p,{'groups.jsonl':10})
    def test_partial_dev_artifacts_and_best_restore(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)
            for name in ['dev/step-800','dev/step-1200','checkpoints/step-800/policy','checkpoints/step-1200/policy','best-adapter']:
                q=p/name;q.mkdir(parents=True);(q/'marker').write_text(name)
            reconcile_artifacts(p,800,800)
            self.assertFalse((p/'dev/step-1200').exists())
            self.assertEqual((p/'best-adapter'/'marker').read_text(),'checkpoints/step-800/policy')
if __name__=='__main__':unittest.main()
