import importlib.util
from pathlib import Path
import sys
import unittest
from unittest.mock import patch
from unittest.mock import Mock
from contextlib import nullcontext
import tempfile

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('restart1600', HERE / 'resume.py')
resume = importlib.util.module_from_spec(spec)
spec.loader.exec_module(resume)


class ResumeTests(unittest.TestCase):
    def test_command_changes_entry_only(self):
        command = ['python', '-u', str(resume.V6 / 'train_grpo_recoverable.py'),
            '--output', '/run', '--resume-checkpoint', '/point', '--evaluation-batch-size', '4']
        actual = resume.wrap_command(command)
        self.assertEqual(actual[4:], command[3:])
        self.assertEqual(actual[3], '--trainer')

    def test_wrong_trainer_is_rejected(self):
        with self.assertRaises(ValueError):
            resume.wrap_command(['python', '-u', '/unexpected.py'])

    def test_real_import_seam_preserves_training_sources_and_applies_both_overrides(self):
        oldpath = sys.path[:]
        oldgate = sys.modules.get('idle_resource_loader')
        original = None
        oldeval = None
        observed = {}
        def run(path, run_name):
            import action_evaluation
            observed.update(path=path, name=run_name, argv=sys.argv[:],
                evaluator=action_evaluation.evaluate_model,
                gate=sys.modules['idle_resource_loader'])
        try:
            sys.path.insert(0, str(resume.ACTION))
            import action_evaluation as original
            oldeval = original.evaluate_model
            with patch.object(resume, 'verify_extension'), patch.object(resume, 'record_policy'), \
                    patch.object(resume, 'model_lease', return_value=nullcontext()), \
                    patch.object(resume, 'assert_no_models'), \
                    patch.object(resume.runpy, 'run_path', run), \
                    patch.object(sys, 'argv', ['resume.py', '--trainer', '--output', '/run', '--resume-checkpoint', '/point']):
                resume.trainer()
            self.assertEqual(observed['path'], str(resume.V6 / 'train_grpo_recoverable.py'))
            self.assertEqual(observed['argv'][1:], ['--output', '/run', '--resume-checkpoint', '/point'])
            self.assertEqual(observed['evaluator'].__module__, 'safe_evaluation')
            import inspect
            self.assertEqual(inspect.signature(observed['gate'].wait_for_idle_training_gpu).parameters['stable_seconds'].default, 60)
        finally:
            sys.path[:] = oldpath
            if original is not None and oldeval is not None:
                original.evaluate_model = oldeval
            if oldgate is None:
                sys.modules.pop('idle_resource_loader', None)
            else:
                sys.modules['idle_resource_loader'] = oldgate

    def test_relative_diagnostic_entry_is_a_project_model(self):
        args = ['python', '-u', 'tasks/action-experiment/diagnosis-dev1600-oom-20261022/history/probe_history.py', '--groups', '25']
        self.assertTrue(resume.is_project_model(args, resume.ACTION.parents[1]))
        self.assertFalse(resume.is_project_model(['python', '/elsewhere/probe.py'], resume.ACTION.parents[1]))

    def test_interrupted_wait_terminates_and_reaps_child_before_propagating(self):
        child = Mock()
        child.pid = 123
        child.poll.return_value = None
        child.wait.side_effect = [KeyboardInterrupt('stop'), 0]
        with self.assertRaises(KeyboardInterrupt):
            resume.ManagedProcess(child).wait()
        child.terminate.assert_called_once_with()
        self.assertEqual(child.wait.call_count, 2)

    def test_child_lease_excludes_second_loader(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'model.lock'
            with resume.model_lease(path):
                with self.assertRaises(BlockingIOError):
                    with resume.model_lease(path):
                        self.fail('second loader acquired model lease')
