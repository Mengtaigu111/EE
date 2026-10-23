"""CPU-only checks for the non-reserving GPU admission policy."""
import builtins
import os
import subprocess
import unittest
from unittest.mock import patch

import idle_resource_loader as gate


class OOM(RuntimeError):
    pass


class IdleGateTests(unittest.TestCase):
    def run_gate(self, query, move=lambda: None, rollback=lambda: None, **kwargs):
        return gate.wait_for_idle_training_gpu(move, rollback, query=query,
            oom_error=OOM, empty_cache=kwargs.pop('empty_cache', lambda: None), stable_seconds=0, **kwargs)

    def test_exact_threshold_loads_without_sleep(self):
        events = []
        self.run_gate(lambda: {'free_mib': 21000, 'foreign_pids': []},
            move=lambda: events.append('move'), on_state=events.append,
            sleep=lambda _: self.fail('no stability window'))
        self.assertEqual([v['status'] for v in events if isinstance(v, dict)], ['loading', 'ready'])
        self.assertIn('move', events)

    def test_foreign_pid_and_low_memory_wait_without_allocating(self):
        snapshots = iter([{'free_mib': 30000, 'foreign_pids': [42]},
                          {'free_mib': 20999, 'foreign_pids': []},
                          {'free_mib': 21000, 'foreign_pids': []}])
        current = [next(snapshots)]
        sleeps, states = [], []
        def sleep(delay):
            sleeps.append(delay)
            current[0] = next(snapshots)
        self.run_gate(lambda: current[0], sleep=sleep, on_state=states.append)
        self.assertEqual(sleeps, [1.0, 1.0])
        self.assertEqual([s['status'] for s in states], ['waiting', 'waiting', 'loading', 'ready'])

    def test_query_error_fails_closed_then_retries(self):
        calls, states = [], []
        def query():
            if not calls:
                calls.append(1)
                raise RuntimeError('bad query')
            return {'free_mib': 22000, 'foreign_pids': []}
        self.run_gate(query, sleep=lambda _: None, on_state=states.append)
        self.assertEqual(states[0]['status'], 'waiting')
        self.assertIn('bad query', states[0]['query_error'])

    def test_malformed_snapshot_fails_closed(self):
        current = [{'free_mib': 'bad', 'foreign_pids': []}]
        states = []
        def sleep(_):
            current[0] = {'free_mib': 21000, 'foreign_pids': []}
        self.run_gate(lambda: current[0], sleep=sleep, on_state=states.append)
        self.assertIn('query_error', states[0])

    def test_oom_rolls_back_and_empties_before_polling(self):
        events = []
        def move():
            events.append('move')
            if events.count('move') == 1:
                raise OOM('race')
        self.run_gate(lambda: {'free_mib': 22000, 'foreign_pids': []}, move=move,
            rollback=lambda: events.append('rollback'), empty_cache=lambda: events.append('empty'),
            sleep=lambda _: events.append('poll'))
        self.assertEqual(events, ['move', 'rollback', 'empty', 'poll', 'move'])

    def test_unrelated_load_error_propagates(self):
        def move():
            raise ValueError('bad model')
        with self.assertRaisesRegex(ValueError, 'bad model'):
            self.run_gate(lambda: {'free_mib': 22000, 'foreign_pids': []}, move=move)

    def test_foreign_process_appearing_during_transfer_rolls_back(self):
        snapshots=iter([{'free_mib':24000,'foreign_pids':[]},
                        {'free_mib':3000,'foreign_pids':[99]},
                        {'free_mib':24000,'foreign_pids':[]},
                        {'free_mib':4000,'foreign_pids':[]}])
        events=[]
        self.run_gate(lambda:next(snapshots),move=lambda:events.append('move'),
            rollback=lambda:events.append('rollback'),empty_cache=lambda:events.append('empty'),
            sleep=lambda _:events.append('poll'))
        self.assertEqual(events,['move','rollback','empty','poll','move'])

    def test_query_interrupt_propagates(self):
        for error in (KeyboardInterrupt, SystemExit):
            with self.assertRaises(error):
                self.run_gate(lambda: (_ for _ in ()).throw(error()))

    def test_waiting_does_not_import_torch(self):
        original_import = builtins.__import__
        def no_torch(name, *args, **kwargs):
            if name == 'torch':
                self.fail('waiting must not import torch or initialize CUDA')
            return original_import(name, *args, **kwargs)
        def stop(_):
            raise KeyboardInterrupt()
        with patch.object(builtins, '__import__', no_torch):
            with self.assertRaises(KeyboardInterrupt):
                gate.wait_for_idle_training_gpu(lambda: self.fail('must wait'), lambda: None,
                    query=lambda: {'free_mib': 1, 'foreign_pids': []}, sleep=stop)

    def test_invalid_policy_rejected(self):
        for kwargs in ({'target_mib': 0}, {'poll_seconds': 0}):
            with self.assertRaises(ValueError):
                self.run_gate(lambda: {}, **kwargs)


class NvidiaQueryTests(unittest.TestCase):
    def runner(self, outputs):
        calls = []
        def run(args, **kwargs):
            calls.append((args, kwargs))
            return subprocess.CompletedProcess(args, 0, next(outputs), '')
        return run, calls

    def test_maps_visible_device_and_excludes_only_own_pid(self):
        run, calls = self.runner(iter(['23000\n', f'{os.getpid()}\n987\n']))
        with patch.dict(os.environ, {'CUDA_VISIBLE_DEVICES': '3,5'}), patch.object(gate.subprocess, 'run', run):
            result = gate.query_gpu_state(device=1)
        self.assertEqual(result, {'free_mib': 23000, 'foreign_pids': [987]})
        for args, kwargs in calls:
            self.assertIn('--id=5', args)
            self.assertGreater(kwargs['timeout'], 0)
            self.assertTrue(kwargs['check'])

    def test_unrestricted_device_uses_requested_index(self):
        run, calls = self.runner(iter(['21000', '']))
        environ = dict(os.environ)
        environ.pop('CUDA_VISIBLE_DEVICES', None)
        with patch.dict(os.environ, environ, clear=True), patch.object(gate.subprocess, 'run', run):
            gate.query_gpu_state(device=2)
        self.assertIn('--id=2', calls[0][0])

    def test_uuid_device_is_preserved(self):
        run, calls = self.runner(iter(['21000\n', '']))
        with patch.dict(os.environ, {'CUDA_VISIBLE_DEVICES': 'GPU-abc'}), patch.object(gate.subprocess, 'run', run):
            self.assertEqual(gate.query_gpu_state()['free_mib'], 21000)
        self.assertIn('--id=GPU-abc', calls[0][0])

    def test_empty_visible_devices_disallows_query(self):
        with patch.dict(os.environ, {'CUDA_VISIBLE_DEVICES': ''}), patch.object(gate.subprocess, 'run') as run:
            with self.assertRaises(ValueError):
                gate.query_gpu_state()
            run.assert_not_called()

    def test_nonzero_timeout_and_bad_output_raise(self):
        for error in (subprocess.CalledProcessError(1, 'nvidia-smi'), subprocess.TimeoutExpired('nvidia-smi', 5)):
            with patch.dict(os.environ, {'CUDA_VISIBLE_DEVICES': '0'}), patch.object(gate.subprocess, 'run', side_effect=error):
                with self.assertRaises(type(error)):
                    gate.query_gpu_state()
        for outputs in (['garbage', ''], ['21000', 'N/A'], ['21000\n22000', '']):
            run, _ = self.runner(iter(outputs))
            with patch.dict(os.environ, {'CUDA_VISIBLE_DEVICES': '0'}), patch.object(gate.subprocess, 'run', run):
                with self.assertRaises(ValueError):
                    gate.query_gpu_state()


if __name__ == '__main__':
    unittest.main()


class FakeClock:
    def __init__(self): self.now = 0.0
    def __call__(self): return self.now
    def sleep(self, seconds): self.now += seconds


class ContinuousIdleTests(unittest.TestCase):
    def run_gate(self, query=None, **kwargs):
        self.clock, self.states, self.moves = FakeClock(), [], []
        return gate.wait_for_idle_training_gpu(
            kwargs.pop('move', lambda: self.moves.append(self.clock())),
            kwargs.pop('rollback', lambda: None),
            query=query or (lambda: {'free_mib': 21000, 'foreign_pids': []}),
            oom_error=OOM, empty_cache=lambda: None, clock=self.clock,
            sleep=self.clock.sleep, on_state=self.states.append, **kwargs)

    def test_default_waits_exactly_sixty_seconds_and_reports_countdown(self):
        self.run_gate()
        self.assertEqual(self.moves, [60.0])
        self.assertEqual(self.clock(), 60.0)
        self.assertEqual(self.states[0]['stable_seconds'], 60)
        self.assertEqual(self.states[0]['idle_elapsed_seconds'], 0)
        self.assertEqual(self.states[59]['idle_elapsed_seconds'], 59)
        self.assertEqual(self.states[-1]['status'], 'ready')
        self.assertEqual(self.states[-1]['idle_elapsed_seconds'], 60)

    def test_low_foreign_and_query_error_each_reset_whole_window(self):
        for condition in ('low', 'foreign', 'query_error'):
            with self.subTest(condition=condition):
                def query():
                    if self.clock() == 30:
                        if condition == 'query_error': raise RuntimeError('temporary unavailable')
                        return {'free_mib': 20999 if condition == 'low' else 21000,
                                'foreign_pids': [33] if condition == 'foreign' else []}
                    return {'free_mib': 21000, 'foreign_pids': []}
                self.run_gate(query=query)
                self.assertEqual(self.moves, [91.0])
                self.assertEqual(self.states[30]['idle_elapsed_seconds'], 0)
                self.assertEqual(self.states[31]['idle_elapsed_seconds'], 0)

    def test_oom_requires_a_new_complete_window_before_retry(self):
        attempts, rollbacks = [], []
        def move():
            attempts.append(self.clock())
            if len(attempts) == 1: raise OOM('transfer race')
        self.run_gate(move=move, rollback=lambda: rollbacks.append(self.clock()))
        self.assertEqual(attempts, [60.0, 121.0])
        self.assertEqual(rollbacks, [60.0])
        self.assertEqual(next(s for s in self.states if s['status'] == 'load_oom')['idle_elapsed_seconds'], 0)

    def test_post_transfer_foreign_and_query_error_require_new_window(self):
        for condition in ('foreign', 'query_error'):
            with self.subTest(condition=condition):
                attempts, rollbacks, loaded = [], [], [False]
                def move():
                    attempts.append(self.clock())
                    loaded[0] = True
                def rollback():
                    rollbacks.append(self.clock())
                    loaded[0] = False
                def query():
                    if loaded[0] and len(attempts) == 1:
                        if condition == 'query_error': raise RuntimeError('query during transfer')
                        return {'free_mib': 1, 'foreign_pids': [33]}
                    return {'free_mib': 1 if loaded[0] else 21000, 'foreign_pids': []}
                self.run_gate(query=query, move=move, rollback=rollback)
                self.assertEqual(attempts, [60.0, 121.0])
                self.assertEqual(rollbacks, [60.0])
                self.assertEqual(self.states[-1]['free_mib'], 1)

    def test_non_integer_poll_reaches_threshold_without_extra_window(self):
        self.run_gate(poll_seconds=7)
        self.assertEqual(self.moves, [63.0])

    def test_zero_seconds_retains_immediate_admission(self):
        self.run_gate(stable_seconds=0)
        self.assertEqual(self.moves, [0.0])

    def test_invalid_stability_seconds_rejected(self):
        for seconds in (-1, float('nan'), float('inf')):
            with self.subTest(seconds=seconds), self.assertRaises(ValueError):
                self.run_gate(stable_seconds=seconds)

    def test_idle_countdown_does_not_import_torch_or_load_model(self):
        original_import = builtins.__import__
        self.clock = FakeClock()
        def no_torch(name, *args, **kwargs):
            if name == 'torch': self.fail('countdown imported torch')
            return original_import(name, *args, **kwargs)
        def stop(delay):
            self.clock.sleep(delay)
            if self.clock() == 59: raise KeyboardInterrupt()
        with patch.object(builtins, '__import__', no_torch), self.assertRaises(KeyboardInterrupt):
            gate.wait_for_idle_training_gpu(lambda: self.fail('loaded too early'), lambda: None,
                query=lambda: {'free_mib': 21000, 'foreign_pids': []}, clock=self.clock, sleep=stop)
