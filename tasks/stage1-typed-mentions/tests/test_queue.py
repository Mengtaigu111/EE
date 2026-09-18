import json
import sys
import tempfile
import unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from run_queue import run_command

class QueueTests(unittest.TestCase):
    def test_completed_stage_is_reused(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d); log=p/'train.log'; output=p/'result.json'
            command=[sys.executable,'-c','import pathlib,sys; pathlib.Path(sys.argv[1]).write_text("ok")',str(output)]
            state={}
            run_command(command,log,state,outputs=[output])
            before=log.read_bytes()
            run_command(command,log,state,outputs=[output])
            self.assertEqual(log.read_bytes(),before)
            self.assertEqual(output.read_text(),'ok')

    def test_failed_attempt_preserved(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d); log=p/'train.log'; output=p/'result.json'; state={}
            import subprocess
            with self.assertRaises(subprocess.CalledProcessError):
                run_command([sys.executable,'-c','import sys;sys.exit(7)'],log,state,outputs=[output])
            run_command([sys.executable,'-c','import pathlib,sys;pathlib.Path(sys.argv[1]).write_text("ok")',str(output)],log,state,outputs=[output])
            exits=list((p/'failed_attempts').glob('*/train.exit.json'))
            self.assertEqual(len(exits),1)
            self.assertEqual(json.loads(exits[0].read_text())['exit_code'],7)

if __name__=='__main__': unittest.main()
