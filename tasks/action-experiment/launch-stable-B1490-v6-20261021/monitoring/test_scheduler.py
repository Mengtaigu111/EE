import json
from pathlib import Path
from subprocess import CompletedProcess, TimeoutExpired
import tempfile
import unittest
from unittest.mock import patch

import scheduler


class FakeClock:
    def __init__(self):
        self.now = 1000.0
        self.sleeps = []

    def time(self):
        return self.now

    def sleep(self, seconds):
        self.sleeps.append(seconds)
        self.now += seconds


class SchedulerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.pipeline = self.root / 'pipeline-state.json'
        self.pipeline.write_text('{"status":"running"}')
        self.state = self.root / 'state.json'
        self.clock = FakeClock()

    def run_scheduler(self, enqueue, steps):
        function = getattr(scheduler, 'run_scheduler', None)
        self.assertTrue(callable(function), 'scheduler loop must be implemented')
        return function(self.state, self.pipeline, enqueue=enqueue,
                        clock=self.clock.time, sleep=self.clock.sleep, steps=steps)

    def test_first_delivery_is_delayed_two_hours_and_cadence_continues(self):
        times = []
        def enqueue():
            times.append(self.clock.now)
            return CompletedProcess([], 0, '', '')
        self.run_scheduler(enqueue, 2)
        self.assertEqual(times, [8200.0, 15400.0])
        state = json.loads(self.state.read_text())
        self.assertEqual(state['next_check_unix'], 22600.0)
        self.assertEqual(state['returncode'], 0)
        self.assertTrue(all(seconds <= 60 for seconds in self.clock.sleeps))

    def test_failed_delivery_retries_in_sixty_seconds(self):
        times = []
        def enqueue():
            times.append(self.clock.now)
            if len(times) == 1:
                return CompletedProcess([], 7, '', 'queue unavailable')
            return CompletedProcess([], 0, '', '')
        self.run_scheduler(enqueue, 1)
        state = json.loads(self.state.read_text())
        self.assertEqual(state['status'], 'delivery_failed')
        self.assertEqual(state['returncode'], 7)
        self.assertEqual(state['stderr'], 'queue unavailable')
        self.assertEqual(state['next_check_unix'], 8260.0)
        self.run_scheduler(enqueue, 1)
        self.assertEqual(times, [8200.0, 8260.0])

    def test_completed_pipeline_exits_only_after_successful_final_delivery(self):
        self.pipeline.write_text('{"status":"completed"}')
        times = []
        def enqueue():
            times.append(self.clock.now)
            return CompletedProcess([], 1 if len(times) == 1 else 0, '', '')
        self.run_scheduler(enqueue, 5)
        self.assertEqual(times, [8200.0, 8260.0])
        state = json.loads(self.state.read_text())
        self.assertEqual(state['status'], 'completed')
        self.assertIsNone(state['next_check_unix'])
        self.run_scheduler(enqueue, 1)
        self.assertEqual(len(times), 2)

    def test_duplicate_instance_is_rejected_without_delivery(self):
        function = getattr(scheduler, 'single_instance', None)
        self.assertTrue(callable(function), 'scheduler needs a process-level lock')
        with function(self.state.with_suffix('.lock')):
            with self.assertRaises(BlockingIOError):
                self.run_scheduler(lambda: self.fail('duplicate delivered'), 1)

    def test_timeout_is_recorded_as_failed_delivery(self):
        def enqueue():
            raise TimeoutExpired(['codex', 'queue'], 60)
        self.run_scheduler(enqueue, 1)
        state = json.loads(self.state.read_text())
        self.assertEqual(state['status'], 'delivery_failed')
        self.assertIsNone(state['returncode'])
        self.assertIsNone(state['last_enqueued_unix'])
        self.assertIn('TimeoutExpired', state['stderr'])
        self.assertEqual(state['next_check_unix'], 8260.0)

    def test_enqueue_uses_safe_argv_and_sixty_second_timeout(self):
        with patch.object(scheduler.subprocess, 'run') as run:
            scheduler.enqueue_check()
        args, kwargs = run.call_args
        self.assertEqual(args[0], [scheduler.CODEX, 'queue', '--thread', scheduler.THREAD,
                                  '--message', scheduler.MESSAGE])
        self.assertEqual(kwargs['timeout'], 60)
        self.assertNotIn('shell', kwargs)
        self.assertEqual(kwargs['cwd'], scheduler.ROOT)

    def test_migrate_preserves_deadline_without_delivery_or_old_state_write(self):
        old = self.root / 'old-state.json'
        old.write_text(json.dumps(dict(status='scheduled', next_check_unix=4321.,
            attempts=12, last_attempt_unix=900., last_enqueued_unix=901.)))
        original = old.read_bytes()
        with patch.object(scheduler, 'enqueue_check') as enqueue:
            state = scheduler.migrate_deadline(old, self.state, clock=self.clock.time)
        enqueue.assert_not_called()
        self.assertEqual(state['next_check_unix'], 4321.)
        self.assertEqual(state['previous_attempts'], 12)
        self.assertEqual(old.read_bytes(), original)
        with self.assertRaises(FileExistsError):
            scheduler.migrate_deadline(old, self.state, clock=self.clock.time)

    def test_message_targets_only_new_b_and_preserves_quiet_notification_rule(self):
        self.assertIn('launch-stable-B1490-v6-20261021', scheduler.MESSAGE)
        self.assertIn('r0-v2-B-stable-v6-from1490-42-20261021', scheduler.MESSAGE)
        self.assertIn('不恢复A', scheduler.MESSAGE)
        self.assertIn('不重启旧B', scheduler.MESSAGE)
        self.assertIn('保持安静', scheduler.MESSAGE)
        self.assertIn('最终完整性检查', scheduler.MESSAGE)


if __name__ == '__main__':
    unittest.main()
