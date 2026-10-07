"""Explicit optimizer-step dev checks using the frozen safe online trainer."""
from numbers import Integral
from pathlib import Path
import sys

from transformers import TrainerCallback

CURRICULUM = Path(__file__).resolve().parents[2] / 'stage2-event-tree/curriculum'
if str(CURRICULUM) not in sys.path:
    sys.path.insert(0, str(CURRICULUM))
from online_dev_trainer import OnlineDevTrainer
from sampling import MicrobatchObservationMixin

EVAL_STEPS = [401, 802, 1203, 1403, 1604, 1804, 2005, 2205, 2406, 2606,
              2807, 3007, 3208]


class ExplicitEvaluationCallback(TrainerCallback):
    """Override periodic flags after DefaultFlowCallback without clearing stop."""
    def __init__(self, checkpoint_steps):
        steps = list(checkpoint_steps)
        if (not steps or any(isinstance(s, bool) or not isinstance(s, Integral) or s <= 0
                             for s in steps) or steps != sorted(set(steps))):
            raise ValueError('Checkpoint steps must be increasing unique positive integers')
        self.steps = tuple(steps)

    def on_train_begin(self, args, state, control, **kwargs):
        if getattr(args, 'eval_on_start', False):
            raise ValueError('Evaluation before training is forbidden')
        if self.steps[-1] != args.max_steps or state.global_step != 0:
            raise ValueError('Fresh training requires explicit checks ending at horizon')
        return control

    def on_step_end(self, args, state, control, **kwargs):
        scheduled = state.global_step in self.steps
        control.should_evaluate = scheduled
        control.should_save = scheduled
        return control


class DenseOnlineDevTrainer(MicrobatchObservationMixin, OnlineDevTrainer):
    def __init__(self, *args, checkpoint_steps, **kwargs):
        super().__init__(*args, **kwargs)
        callback = ExplicitEvaluationCallback(checkpoint_steps)
        self.checkpoint_steps = callback.steps
        self.add_callback(callback)

    def evaluate(self, *args, **kwargs):
        if self.state.global_step not in self.checkpoint_steps:
            raise ValueError('Dev evaluation outside explicit checkpoint steps')
        return super().evaluate(*args, **kwargs)
