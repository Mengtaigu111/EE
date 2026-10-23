import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

from checkpoint_state import load_checkpoint, save_checkpoint
import recovery_runtime
import run_pipeline


class PipelineRecoveryTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.output = self.root / 'smoke'
        self.output.mkdir()

    def metrics(self, initial='a' * 64, final='b' * 64, reference=True):
        value = dict(status='completed', initial_policy_parameter_sha256=initial,
                     final_policy_parameter_sha256=final, reference_unchanged=reference)
        (self.output / 'train_metrics.json').write_text(json.dumps(value))
        return value

    def validate(self):
        function = getattr(recovery_runtime, 'validate_smoke', None)
        self.assertTrue(callable(function), 'Smoke validation must survive checkpoint pruning')
        return function(self.output)

    def archive(self):
        function = getattr(recovery_runtime, 'archive_uninitialized_run', None)
        self.assertTrue(callable(function), 'Initialization failures need a safe restart path')
        return function(self.output)

    def mark_initializing(self):
        (self.output / 'initialization.json').write_text(json.dumps(
            {'format': 'grpo-initialization-v1', 'status': 'initializing'}))

    def checkpoint_model(self):
        model = torch.nn.Linear(2, 1)
        return model, torch.optim.AdamW(model.parameters())

    def pipeline(self, completed=True):
        launch = self.root / 'launch'
        launch.mkdir()
        (launch / 'source-sha256.json').write_text('{}')
        with patch.object(run_pipeline, 'LAUNCH', launch), \
                patch.object(run_pipeline, 'stages', return_value=[('smoke-B', self.output, [])]), \
                patch.dict('os.environ', {'TMUX': 'cpu-test'}), \
                patch.object(run_pipeline.subprocess, 'run') as process:
            if not completed:
                def complete(*args, **kwargs):
                    self.output.mkdir(exist_ok=True)
                    self.metrics()
                    return SimpleNamespace(returncode=0)
                process.side_effect = complete
            run_pipeline.main()
            return process.call_count

    def test_smoke_survives_initial_checkpoint_pruning(self):
        model, optimizer = self.checkpoint_model()
        points = self.output / 'recovery-checkpoints'
        for name in ('initial-0', 'dev-0', 'step-1', 'final-2'):
            save_checkpoint(points, name, model, optimizer, {})
        recovery_runtime.prune_checkpoints(points)
        self.assertFalse((points / 'initial-0').exists())
        self.metrics()
        self.validate()

    def test_smoke_uses_original_digest_restored_in_checkpoint_payload(self):
        model, optimizer = self.checkpoint_model()
        initial = hashlib.sha256(model.weight.detach().numpy().tobytes()).hexdigest()
        checkpoint = save_checkpoint(self.output / 'recovery-checkpoints', 'initial-0',
            model, optimizer, {'initial_policy_parameter_sha256': initial})
        payload = load_checkpoint(checkpoint, model, optimizer)
        with torch.no_grad():
            model.weight.add_(1)
        final = hashlib.sha256(model.weight.detach().numpy().tobytes()).hexdigest()
        self.metrics(initial=payload['initial_policy_parameter_sha256'], final=final)
        self.validate()

    def test_smoke_rejects_missing_or_invalid_digests_and_non_boolean_reference(self):
        cases = [('a' * 64, 'a' * 64, True), (None, 'b' * 64, True),
                 ('a' * 63, 'b' * 64, True), ('g' * 64, 'b' * 64, True),
                 ('a' * 64, 'b' * 64, 1), ('a' * 64, 'b' * 64, False)]
        for initial, final, reference in cases:
            with self.subTest(initial=initial, final=final, reference=reference):
                self.metrics(initial, final, reference)
                with self.assertRaises(ValueError):
                    self.validate()

    def test_completed_smoke_is_revalidated_before_skipping(self):
        self.metrics(initial='a' * 64, final='a' * 64)
        with self.assertRaises(ValueError):
            self.pipeline()

    def test_valid_completed_smoke_is_skipped_without_training(self):
        self.metrics()
        self.assertEqual(self.pipeline(), 0)

    def test_fresh_smoke_validation_does_not_need_initial_adapter(self):
        self.assertEqual(self.pipeline(completed=False), 1)

    def test_initial_export_failure_with_inherited_groups_can_restart(self):
        self.mark_initializing()
        (self.output / 'groups.jsonl').write_text('{"group_index":0,"inherited":true}\n')
        model, optimizer = self.checkpoint_model()
        def fail_export(model, path):
            path.mkdir()
            raise RuntimeError('initial adapter export failed')
        with self.assertRaisesRegex(RuntimeError, 'initial adapter export failed'):
            save_checkpoint(self.output / 'recovery-checkpoints', 'initial-800',
                            model, optimizer, {}, export_policy=fail_export)
        archived = self.archive()
        self.assertFalse(self.output.exists())
        self.assertEqual((archived / 'groups.jsonl').read_text(),
                         '{"group_index":0,"inherited":true}\n')

    def test_unknown_training_logs_refuse_archive(self):
        (self.output / 'groups.jsonl').write_text('{"group_index":1}\n')
        with self.assertRaises(RuntimeError):
            self.archive()
        self.assertTrue(self.output.exists())

    def test_invalid_initializing_marker_cannot_authorize_archive(self):
        self.mark_initializing()
        (self.output / 'initialization.json').write_text('{"status":"initializing"}')
        (self.output / 'dev-history.jsonl').write_text('{}\n')
        with self.assertRaises(RuntimeError):
            self.archive()
        self.assertTrue(self.output.exists())

    def test_published_checkpoint_without_latest_refuses_archive(self):
        self.mark_initializing()
        model, optimizer = self.checkpoint_model()
        points = self.output / 'recovery-checkpoints'
        save_checkpoint(points, 'initial-800', model, optimizer, {})
        (points / 'latest.json').unlink()
        with self.assertRaises(RuntimeError):
            self.archive()
        self.assertTrue(self.output.exists())

    def test_existing_latest_refuses_archive_even_with_initializing_marker(self):
        self.mark_initializing()
        points = self.output / 'recovery-checkpoints'
        points.mkdir()
        (points / 'latest.json').write_text('{}')
        with self.assertRaises(RuntimeError):
            self.archive()

    def test_pretraining_output_without_logs_can_restart(self):
        (self.output / 'configuration.json').write_text('{}')
        archived = self.archive()
        self.assertFalse(self.output.exists())
        self.assertTrue((archived / 'configuration.json').is_file())


if __name__ == '__main__':
    unittest.main()
