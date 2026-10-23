"""Configurable Action SFT TC stopping preserves the historical default."""
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from train_sft import TCFeedbackController, parser


def test_cli_default_six_and_explicit_three():
    assert parser().parse_args(['--output', 'unused']).early_stopping_patience == 6
    assert parser().parse_args(['--output', 'unused', '--early-stopping-patience', '3']).early_stopping_patience == 3


@pytest.mark.parametrize('value', [0, -1, True, False, 1.5, '3', None])
def test_controller_rejects_nonpositive_or_noninteger_patience(value):
    with pytest.raises(ValueError, match='patience'):
        TCFeedbackController(patience=value)


def test_patience_three_stops_on_third_no_gain_without_reducing_lr():
    controller = TCFeedbackController(patience=3)
    assert controller.state_dict()['patience'] == 3
    first = controller.update(200, dict(TC_f1=.6, TI_f1=.7))
    assert first['bad_count'] == 0 and not first['stop']
    for step in (400, 600):
        decision = controller.update(step, dict(TC_f1=.6, TI_f1=.8))
        assert not decision['stop'] and not decision['reduce_lr']
    decision = controller.update(800, dict(TC_f1=.6005, TI_f1=.9))
    assert decision['bad_count'] == 3 and decision['stop'] and not decision['reduce_lr']
    assert decision['best_step'] == 800


def test_effective_gain_resets_patience_three():
    controller = TCFeedbackController(patience=3)
    controller.update(1, dict(TC_f1=.6, TI_f1=.7))
    controller.update(2, dict(TC_f1=.6, TI_f1=.7))
    controller.update(3, dict(TC_f1=.6, TI_f1=.7))
    decision = controller.update(4, dict(TC_f1=.601, TI_f1=.7))
    assert decision['bad_count'] == 0 and not decision['stop']
    for step in (5, 6):
        assert not controller.update(step, dict(TC_f1=.601, TI_f1=.7))['stop']
    assert controller.update(7, dict(TC_f1=.601, TI_f1=.7))['stop']


def test_historical_six_reduces_once_on_three_and_stops_on_six():
    controller = TCFeedbackController()
    controller.update(1, dict(TC_f1=.6, TI_f1=.7))
    decisions = [controller.update(step, dict(TC_f1=.6, TI_f1=.7)) for step in range(2, 8)]
    assert [d['bad_count'] for d in decisions if d['reduce_lr']] == [3]
    assert [d['bad_count'] for d in decisions if d['stop']] == [6]
    assert controller.state_dict()['patience'] == 6


def test_callback_patience_three_stops_without_scheduler_reduction(tmp_path):
    from types import SimpleNamespace
    from train_sft import make_dev_callback
    class Trainer:
        reductions = 0
        def save_model(self, path):
            pass
        def plateau_scheduler(self):
            return self
        def reduce_lr(self):
            self.reductions += 1
    class Tokenizer:
        def save_pretrained(self, path):
            pass
    trainer = Trainer()
    callback = make_dev_callback(trainer, lambda model, step: {'metrics': {'TC_f1': .6, 'TI_f1': .7}},
                                 tmp_path, Tokenizer(), [1, 2, 3, 4], controller=TCFeedbackController(patience=3))
    trainer.model = object()
    control = SimpleNamespace(should_training_stop=False)
    for step in (1, 2, 3):
        callback.on_step_end(None, SimpleNamespace(global_step=step), control)
        assert not control.should_training_stop
    callback.on_step_end(None, SimpleNamespace(global_step=4), control)
    assert control.should_training_stop and trainer.reductions == 0
    assert callback.feedback.state_dict()['patience'] == 3


@pytest.mark.parametrize('value', ['0', '-1'])
def test_main_rejects_invalid_cli_patience_before_loading_gpu(monkeypatch, value):
    from train_sft import main
    monkeypatch.setattr(sys, 'argv', ['train_sft.py', '--output', 'unused', '--early-stopping-patience', value])
    with pytest.raises(ValueError, match='early-stopping-patience'):
        main()
