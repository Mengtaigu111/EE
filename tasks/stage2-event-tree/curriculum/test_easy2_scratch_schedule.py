"""CPU acceptance test for scratch two epochs followed by unequal full-data phase."""
import tempfile
import unittest
from pathlib import Path

import torch

from test_curriculum_trainer import curriculum, trainer


class ScratchEasyTwoEpochTests(unittest.TestCase):
    def test_nondivisible_scratch_two_epochs_and_changed_data_resume(self):
        torch.set_num_threads(1)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            first = trainer(root / 'stage1', stop=6, size=5)
            first.args.max_steps = 9
            trained_first = []
            first_step = first.training_step

            def record_first(model, inputs, num_items_in_batch=None):
                trained_first.extend(inputs['input_ids'][:, 1].tolist())
                return first_step(model, inputs, num_items_in_batch)

            first.training_step = record_first
            first.train()
            self.assertEqual(first.state.global_step, 6)
            self.assertEqual(first.state.epoch, 2.0)
            self.assertEqual(len(trained_first), 10)
            self.assertEqual(sorted(trained_first[:5]), list(range(2, 7)))
            self.assertEqual(sorted(trained_first[5:]), list(range(2, 7)))
            boundary = root / 'stage1' / 'checkpoint-6'
            curriculum.validate_resume_checkpoint(boundary, expected_step=6,
                                                  expected_horizon=9)
            self.assertEqual(first.lr_scheduler.state_dict()['last_epoch'], 6)
            boundary_lr = first.optimizer.param_groups[0]['lr']
            self.assertGreater(boundary_lr, 0)

            second = trainer(root / 'stage2', stop=9, size=7,
                             resume_new_data=True)
            second.args.max_steps = 9
            trained_second = []
            second_step = second.training_step

            def record_second(model, inputs, num_items_in_batch=None):
                trained_second.extend(inputs['input_ids'][:, 1].tolist())
                return second_step(model, inputs, num_items_in_batch)

            second.training_step = record_second
            second.train(resume_from_checkpoint=str(boundary))
            self.assertEqual(second.state.global_step, 9)
            self.assertEqual(second.lr_scheduler.state_dict()['last_epoch'], 9)
            self.assertEqual(len(trained_second), 6)
            self.assertEqual(len(set(trained_second)), 6)
            evidence = second.resume_evidence
            for field in ('model_exact', 'optimizer_exact', 'scheduler_exact',
                          'rng_exact'):
                self.assertIs(evidence[field], True)
            self.assertEqual(evidence['global_step'], 6)
            self.assertEqual(evidence['horizon'], 9)
            self.assertIs(evidence['data_skip'], False)
            self.assertEqual(evidence['learning_rates'], [boundary_lr, boundary_lr])
            losses = [entry['step'] for entry in second.state.log_history
                      if 'loss' in entry]
            self.assertEqual(losses, list(range(1, 10)))
            curriculum.validate_resume_checkpoint(root / 'stage2' / 'checkpoint-9',
                                                  expected_step=9,
                                                  expected_horizon=9)


if __name__ == '__main__':
    unittest.main()
