import json
import random
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path

import torch
from checkpoint_state import load_checkpoint, save_checkpoint, resolve_latest


class CheckpointTests(unittest.TestCase):
    def model_optimizer(self):
        model = torch.nn.Linear(3, 2)
        return model, torch.optim.AdamW(model.parameters(), lr=0.01)

    def export(self, model, path):
        path.mkdir()
        torch.save(model.state_dict(), path / 'weights.pt')

    def restore(self, model, path):
        model.load_state_dict(torch.load(path / 'weights.pt', weights_only=True))

    def test_pending_gradient_resume_exact_update_and_rng(self):
        random.seed(31)
        torch.manual_seed(31)
        model, optimizer = self.model_optimizer()
        x = torch.randn(4, 3)
        model(x).square().sum().backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        model(x).sum().backward()  # A pending accumulation must survive restart.
        payload = {'loop': {'step': 1, 'next_group': 5, 'pending': 1},
                   'guard': {'window': [torch.tensor([0.2, 0.3])]},
                   'earlystop': {'bad': 2}}
        with tempfile.TemporaryDirectory() as directory:
            path = save_checkpoint(directory, 'step-1', model, optimizer, payload, self.export)
            expected_random = random.random(), torch.rand(4)
            model(x).square().sum().backward()
            optimizer.step()
            expected = {k: v.clone() for k, v in model.state_dict().items()}
            restored, restored_optimizer = self.model_optimizer()
            actual = load_checkpoint(path, restored, restored_optimizer, self.restore)
            self.assertEqual(actual['loop'], payload['loop'])
            self.assertTrue(torch.equal(actual['guard']['window'][0], payload['guard']['window'][0]))
            self.assertEqual(random.random(), expected_random[0])
            self.assertTrue(torch.equal(torch.rand(4), expected_random[1]))
            restored(x).square().sum().backward()
            restored_optimizer.step()
            for key, value in restored.state_dict().items():
                self.assertTrue(torch.equal(value, expected[key]), key)

    def test_tampering_rejected_before_policy_loading(self):
        model, optimizer = self.model_optimizer()
        with tempfile.TemporaryDirectory() as directory:
            path = save_checkpoint(directory, 'step-1', model, optimizer, {}, self.export)
            with (path / 'adapter' / 'weights.pt').open('ab') as stream:
                stream.write(b'corrupt')
            with self.assertRaisesRegex(ValueError, 'hash'):
                load_checkpoint(path, model, optimizer, self.restore)

    def test_failed_export_preserves_latest(self):
        model, optimizer = self.model_optimizer()
        with tempfile.TemporaryDirectory() as directory:
            first = save_checkpoint(directory, 'step-1', model, optimizer, {}, self.export)
            def broken(model, path):
                path.mkdir()
                (path / 'partial').write_text('unfinished')
                raise RuntimeError('disk failure')
            with self.assertRaisesRegex(RuntimeError, 'disk failure'):
                save_checkpoint(directory, 'step-2', model, optimizer, {}, broken)
            self.assertEqual(resolve_latest(directory), first)
            self.assertFalse((Path(directory) / 'step-2').exists())
            load_checkpoint(first, model, optimizer, self.restore)

    def test_failed_pointer_publish_preserves_latest(self):
        model, optimizer = self.model_optimizer()
        with tempfile.TemporaryDirectory() as directory:
            first = save_checkpoint(directory, 'step-1', model, optimizer, {}, self.export)
            with patch('checkpoint_state.os.replace', side_effect=OSError('publish failed')):
                with self.assertRaisesRegex(OSError, 'publish failed'):
                    save_checkpoint(directory, 'step-2', model, optimizer, {}, self.export)
            self.assertEqual(resolve_latest(directory), first)
            load_checkpoint(first, model, optimizer, self.restore)

    def test_parameter_names_must_match(self):
        model, optimizer = self.model_optimizer()
        with tempfile.TemporaryDirectory() as directory:
            path = save_checkpoint(directory, 'step-1', model, optimizer, {}, self.export)
            other = torch.nn.Sequential(torch.nn.Linear(3, 2))
            with self.assertRaisesRegex(ValueError, 'parameter'):
                load_checkpoint(path, other, torch.optim.AdamW(other.parameters()))


if __name__ == '__main__':
    unittest.main()
