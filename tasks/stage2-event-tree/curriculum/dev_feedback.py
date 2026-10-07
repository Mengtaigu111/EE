"""Serializable development feedback and optimizer-update LR scheduling.

Checkpoint ranking uses raw scores; plateau decisions use a separate anchor.
The scheduler counts optimizer updates, including throughout warmup, and leaves
optimizer checkpointing to the training framework.
"""
import math
from numbers import Integral, Real

from torch.optim.lr_scheduler import LRScheduler


class FeedbackController:
    """Rank every score while requiring ``min_delta`` to reset patience."""

    def __init__(self, min_delta=.001):
        if (isinstance(min_delta, bool) or not isinstance(min_delta, Real)
                or not math.isfinite(min_delta) or min_delta <= 0):
            raise ValueError('min_delta must be a positive finite number')
        self.min_delta = float(min_delta)
        self.best_score = None
        self.best_step = None
        self.anchor_score = None
        self.bad_count = 0
        self.eval_count = 0
        self.last_eval_step = None

    @property
    def last_step(self):
        return self.last_eval_step

    def update(self, step, score):
        """Return decisions and resulting state after one increasing eval step."""
        if isinstance(step, bool) or not isinstance(step, Integral) or step <= 0:
            raise ValueError('step must be a positive integer')
        if self.last_eval_step is not None and step <= self.last_eval_step:
            raise ValueError('evaluation steps must be strictly increasing')
        if (isinstance(score, bool) or not isinstance(score, Real)
                or not math.isfinite(score) or not 0 <= score <= 1):
            raise ValueError('score must be a finite number in [0, 1]')
        step, score = int(step), float(score)
        new_best = self.best_score is None or score >= self.best_score
        if new_best:
            self.best_score, self.best_step = score, step
        if self.anchor_score is None or score - self.anchor_score >= self.min_delta - 1e-12:
            self.anchor_score = score
            self.bad_count = 0
        else:
            self.bad_count += 1
        self.eval_count += 1
        self.last_eval_step = step
        stop = self.bad_count >= 6
        return dict(self.state_dict(), new_best=new_best,
                    reduce_lr=self.bad_count == 3 and not stop, stop=stop)

    def state_dict(self):
        return {
            'min_delta': self.min_delta,
            'best_score': self.best_score,
            'best_step': self.best_step,
            'anchor_score': self.anchor_score,
            'bad_count': self.bad_count,
            'eval_count': self.eval_count,
            'last_eval_step': self.last_eval_step,
        }

    def load_state_dict(self, state_dict):
        """Restore an independent copy of the scalar checkpoint state."""
        for field in self.state_dict():
            setattr(self, field, state_dict[field])


class WarmupPlateauScheduler(LRScheduler):
    """Linear update-count warmup multiplied by a persistent plateau scale.

    Construction performs PyTorch's initial step: ``last_epoch == 0`` and LR
    is zero, matching HF LambdaLR warmup. Call ``step()`` after each optimizer
    update. Accelerate can wrap this standard LRScheduler; plateau events use
    ``wrapped.scheduler.reduce_lr()`` when a wrapper is present.
    """

    def __init__(self, optimizer, warmup_steps=97):
        if (isinstance(warmup_steps, bool) or not isinstance(warmup_steps, Integral)
                or warmup_steps <= 0):
            raise ValueError('warmup_steps must be a positive integer')
        self.warmup_steps = int(warmup_steps)
        self.plateau_scale = 1.0
        super().__init__(optimizer)

    def get_lr(self):
        warmup_scale = min(self.last_epoch / self.warmup_steps, 1.0)
        return [base_lr * warmup_scale * self.plateau_scale for base_lr in self.base_lrs]

    def _sync_lr(self):
        self._last_lr = self.get_lr()
        for group, learning_rate in zip(self.optimizer.param_groups, self._last_lr):
            group['lr'] = learning_rate

    def reduce_lr(self):
        """Halve all current and future rates without advancing update count."""
        self.plateau_scale *= .5
        self._sync_lr()

    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict)
        self._sync_lr()
