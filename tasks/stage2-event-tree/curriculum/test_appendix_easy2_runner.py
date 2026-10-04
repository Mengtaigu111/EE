"""CPU checks for same-prompt curriculum and fixed evaluation adapter."""
import argparse
import importlib
import json
from pathlib import Path
import sys
import shutil
import subprocess
import tempfile
import unittest
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

class AppendixEasy2Tests(unittest.TestCase):
    def test_schedule(self):
        runner = importlib.import_module('run_appendix_easy2_curriculum')
        self.assertEqual(runner.verify_budget(4010, 814, 4010), (814, 3196))
        self.assertEqual(runner.checkpoint_steps(4010, 814, 802), [802, 814, 1604, 2406, 3208, 4010])
        self.assertEqual(runner.smoke_schedule(), (6, 2, 2))
        manifest = dict(stage1_total_count=6504, stage1_epoch_target=2,
                        stage1_optimizer_steps=814, stage2_optimizer_steps=3196)
        self.assertEqual(runner.validate_schedule(manifest, dict(microbatch=1, gradient_accumulation=16), 4010), 814)
        with self.assertRaises(ValueError):
            runner.verify_budget(4010, 814, 4009)

    def test_fixed_eval_commands_and_gate(self):
        adapter = importlib.import_module('appendix_fixed_eval')
        commands = []
        args = argparse.Namespace(data_dir=Path('/sentence'), document_data_dir=Path('/document'),
                                  model='model', seed=42, batch_size=4)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            metric = root / 'checkpoint-eval/test-checkpoint-2.metrics.json'
            metric.parent.mkdir()
            metric.write_text(json.dumps({'complex': {'exit_code': 0,
                'scores': {k: 0.0 for k in adapter.COMPLEX_KEYS}}}))
            with patch.object(adapter, 'run_stage', side_effect=lambda cmd, log: commands.append(cmd)):
                adapter.evaluate_checkpoint(Path('/checkpoint-2'), 'test', args, root, 100)
            self.assertEqual(Path(commands[0][1]).name, 'predict_appendix_fixed.py')
            self.assertEqual(Path(commands[1][1]).parent, adapter.FIXED)
            self.assertEqual(Path(commands[2][1]).parent, adapter.FIXED)
            self.assertIn('--complex', commands[2])
            metric.write_text(json.dumps({'complex': {'exit_code': 1, 'scores': {}}}))
            with patch.object(adapter, 'run_stage'):
                with self.assertRaises(ValueError):
                    adapter.evaluate_checkpoint(Path('/checkpoint-2'), 'test', args, root, 100)

    def test_corrected_baseline_selection(self):
        runner = importlib.import_module('run_appendix_easy2_curriculum')
        paths = runner.corrected_baseline_paths()
        self.assertEqual(paths['step'], 2406)
        self.assertAlmostEqual(runner.read(paths['dev'])['official']['AC_f1'], 0.4126373626373626)
        self.assertAlmostEqual(runner.read(paths['test'])['official']['AC_f1'], 0.41879652049143573)

    def test_wrapper_imports_repaired_parser_in_fresh_process(self):
        code = "import sys;sys.path.insert(0,sys.argv[1]);import predict_appendix_fixed as w;w.runpy.run_path=lambda *a,**k: __import__('linearize');w.main();import linearize;print(linearize.__file__);linearize.parse_forest('(ROOT (FAC (Trigger ())))')"
        result = subprocess.run([sys.executable, '-c', code, str(HERE)], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('evaluation-v2-20261003/scripts/linearize.py', result.stdout)

    def test_report_uses_corrected_same_prompt_baseline(self):
        runner = importlib.import_module('run_appendix_easy2_curriculum')
        old = HERE.parent / 'runs/C-CL-Mean-UserV2-Easy2-42-20261002'
        state = runner.read(old / 'run.json')
        fixed = runner.corrected_baseline_paths()
        state['baseline_corrected_dev_metrics'] = fixed['dev']
        state['baseline_corrected_test_metrics'] = fixed['test']
        manifest = runner.read(HERE / 'appendix-easy2-20261003/mean_curriculum_manifest.json')
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory)
            for phase in ('stage1', 'stage2'):
                (target / phase).mkdir()
                shutil.copy(old / phase / 'train_metrics.json', target / phase / 'train_metrics.json')
            shutil.copy(old / 'stage2/resume_evidence.json', target / 'stage2/resume_evidence.json')
            runner.report(target, state, manifest, publish=False)
            text = (target / 'mean_curriculum_training_report.md').read_text()
            self.assertIn('41.2637', text)
            self.assertIn('41.8797', text)
            self.assertIn('Stage1=814; Stage2=3196', text)
            self.assertNotIn('prompt 与 curriculum 同时变化', text)
            self.assertIn('两组 prompt 与训练预算相同', text)


if __name__ == '__main__':
    unittest.main()
