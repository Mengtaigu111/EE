"""CPU contract tests for dev feedback and optimizer-update warmup."""
import copy
import math
import unittest

import torch

try:
    import dev_feedback
except ModuleNotFoundError:
    dev_feedback = None


class FeedbackTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(dev_feedback, 'dev_feedback implementation is missing')
        self.controller = dev_feedback.FeedbackController()

    def test_initial_state_and_exact_delta(self):
        initial = self.controller.state_dict()
        self.assertIsNone(initial['best_score'])
        self.assertIsNone(initial['anchor_score'])
        self.assertIsNone(initial['best_step'])
        self.assertEqual(initial['eval_count'], 0)
        first = self.controller.update(50, .7)
        self.assertTrue(first['new_best'])
        self.assertEqual(first['bad_count'], 0)
        result = self.controller.update(100, .701)
        self.assertTrue(result['new_best'])
        self.assertEqual(result['bad_count'], 0)
        self.assertEqual(result['anchor_score'], .701)
        self.assertEqual(result['eval_count'], 2)
        self.assertEqual(result['last_eval_step'], 100)

    def test_tiny_improvement_and_tie_update_raw_best_but_not_anchor(self):
        self.controller.update(1, .7)
        tiny = self.controller.update(2, .7004)
        tied = self.controller.update(3, .7004)
        self.assertTrue(tiny['new_best'])
        self.assertTrue(tied['new_best'])
        self.assertEqual(tied['best_step'], 3)
        self.assertEqual(tied['best_score'], .7004)
        self.assertEqual(tied['anchor_score'], .7)
        self.assertEqual(tied['bad_count'], 2)
        improved = self.controller.update(4, .701)
        self.assertEqual(improved['bad_count'], 0)

    def test_plateau_reduction_stop_priority_and_reset_cycle(self):
        self.controller.update(1, .5)
        for step in range(2, 8):
            result = self.controller.update(step, .5)
            self.assertEqual(result['bad_count'], step - 1)
            self.assertEqual(result['reduce_lr'], step == 4)
            self.assertEqual(result['stop'], step == 7)
        self.assertTrue(self.controller.update(8, .4)['stop'])
        resumed = self.controller.update(9, .501)
        self.assertFalse(resumed['stop'])
        self.assertEqual(resumed['bad_count'], 0)
        for step in range(10, 13):
            result = self.controller.update(step, .501)
        self.assertTrue(result['reduce_lr'])

    def test_reject_malformed_values_without_state_mutation(self):
        self.controller.update(50, .5)
        state = self.controller.state_dict()
        for score in [float('nan'), float('inf'), -float('inf'), -.01, 1.01, None, '0.5', True]:
            with self.subTest(score=score), self.assertRaises((ValueError, TypeError)):
                self.controller.update(100, score)
            self.assertEqual(self.controller.state_dict(), state)
        for step in [50, 49, -1, 0, 1.5, '100', True, None]:
            with self.subTest(step=step), self.assertRaises((ValueError, TypeError)):
                self.controller.update(step, .6)
            self.assertEqual(self.controller.state_dict(), state)

    def test_feedback_state_restore_continues_same_decisions(self):
        self.controller.update(10, .6)
        self.controller.update(20, .6004)
        other = dev_feedback.FeedbackController(min_delta=.2)
        other.load_state_dict(copy.deepcopy(self.controller.state_dict()))
        self.assertEqual(other.state_dict(), self.controller.state_dict())
        for step, score in [(30, .6), (40, .6), (50, .601), (60, .6)]:
            self.assertEqual(other.update(step, score), self.controller.update(step, score))


class SchedulerTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(dev_feedback, 'dev_feedback implementation is missing')

    def make_optimizer(self):
        parameter = torch.nn.Parameter(torch.tensor(1.))
        return torch.optim.SGD([{'params': [parameter], 'lr': .01},
                                {'params': [torch.nn.Parameter(torch.tensor(2.))], 'lr': .02}])

    def test_update_warmup_and_immediate_plateau_reduction(self):
        optimizer = self.make_optimizer()
        scheduler = dev_feedback.WarmupPlateauScheduler(optimizer)
        self.assertIsInstance(scheduler, torch.optim.lr_scheduler.LRScheduler)
        self.assertEqual(scheduler.last_epoch, 0)
        self.assertEqual(scheduler.get_last_lr(), [0., 0.])
        for step in range(1, 102):
            optimizer.step()
            scheduler.step()
            if step in [10, 97]:
                epoch = scheduler.last_epoch
                scheduler.reduce_lr()
                self.assertEqual(scheduler.last_epoch, epoch)
            scale = .5 ** ((step >= 10) + (step >= 97))
            expected = [lr * min(step / 97, 1.) * scale for lr in [.01, .02]]
            self.assertEqual(scheduler.get_last_lr(), expected)
            self.assertEqual([g['lr'] for g in optimizer.param_groups], expected)

    def test_state_restore_owns_no_optimizer_and_continues_exactly(self):
        optimizer = self.make_optimizer()
        scheduler = dev_feedback.WarmupPlateauScheduler(optimizer, warmup_steps=7)
        for _ in range(3):
            optimizer.step()
            scheduler.step()
        scheduler.reduce_lr()
        state = copy.deepcopy(scheduler.state_dict())
        self.assertNotIn('optimizer', state)
        restored_optimizer = self.make_optimizer()
        restored_scheduler = dev_feedback.WarmupPlateauScheduler(restored_optimizer)
        restored_optimizer.load_state_dict(copy.deepcopy(optimizer.state_dict()))
        restored_scheduler.load_state_dict(state)
        self.assertEqual(restored_scheduler.state_dict(), scheduler.state_dict())
        self.assertEqual(restored_scheduler.get_last_lr(), scheduler.get_last_lr())
        for step in range(8):
            optimizer.step()
            restored_optimizer.step()
            scheduler.step()
            restored_scheduler.step()
            if step == 2:
                scheduler.reduce_lr()
                restored_scheduler.reduce_lr()
            self.assertEqual(restored_scheduler.state_dict(), scheduler.state_dict())
            self.assertEqual([g['lr'] for g in restored_optimizer.param_groups], scheduler.get_last_lr())

    def test_smoke_warmup_one_and_accelerate_wrapper(self):
        from accelerate import Accelerator
        accelerator = Accelerator(cpu=True)
        optimizer = self.make_optimizer()
        scheduler = dev_feedback.WarmupPlateauScheduler(optimizer, warmup_steps=math.ceil(4 * .03))
        wrapped_optimizer, wrapped_scheduler = accelerator.prepare(optimizer, scheduler)
        wrapped_optimizer.step()
        wrapped_scheduler.step()
        self.assertEqual(scheduler.last_epoch, 1)
        self.assertEqual(wrapped_scheduler.get_last_lr(), [.01, .02])
        wrapped_scheduler.scheduler.reduce_lr()
        self.assertEqual(wrapped_scheduler.get_last_lr(), [.005, .01])
        state = copy.deepcopy(wrapped_scheduler.state_dict())
        wrapped_scheduler.load_state_dict(state)
        self.assertEqual(wrapped_scheduler.state_dict(), state)

    def test_invalid_warmup(self):
        for warmup in [0, -1, 1.5, True, None]:
            with self.subTest(warmup=warmup), self.assertRaises((ValueError, TypeError)):
                dev_feedback.WarmupPlateauScheduler(self.make_optimizer(), warmup_steps=warmup)


if __name__ == '__main__':
    unittest.main()
