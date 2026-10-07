"""Resident-model online dev feedback with complete safe checkpoints."""
import json
import math
from pathlib import Path
import time

from curriculum_trainer import _hash, _write_json, validate_resume_checkpoint
from dev_feedback import FeedbackController, WarmupPlateauScheduler
from sorted_first_sampler import SortedFirstTrainer

OFFICIAL_KEYS = tuple(f'{metric}_{part}'
                     for metric in ('TI', 'TC', 'AI', 'AC', 'EC_I', 'EC_C')
                     for part in ('p', 'r', 'f1'))


def validate_online_checkpoint(path, expected_step=None, expected_horizon=None):
    """Require feedback state as well as the inherited full-state integrity seal."""
    path = Path(path)
    state = validate_resume_checkpoint(path, expected_step, expected_horizon)
    manifest = json.loads((path/'continuity.json').read_text())
    if 'feedback_state.json' not in manifest['sha256']:
        raise ValueError('Online checkpoint has no sealed feedback state')
    feedback = json.loads((path/'feedback_state.json').read_text())
    if feedback.get('format') != 'online-dev-feedback-v1':
        raise ValueError('Unknown feedback state format')
    controller = FeedbackController()
    controller.load_state_dict(feedback['controller'])
    if controller.last_step is not None and controller.last_step > state['global_step']:
        raise ValueError('Feedback step exceeds checkpoint step')
    return state


class OnlineDevTrainer(SortedFirstTrainer):
    def __init__(self, *args, dev_evaluator, feedback_path, warmup_steps=97, **kwargs):
        self.dev_evaluator = dev_evaluator
        self.feedback_path = Path(feedback_path)
        self.feedback_path.parent.mkdir(parents=True, exist_ok=True)
        # Distinct files on resume keep inherited sampling evidence exclusive.
        with self.feedback_path.open('x', encoding='utf-8'):
            pass
        self.feedback_controller = FeedbackController()
        self.feedback_warmup_steps = warmup_steps
        self._feedback_new_best = False
        super().__init__(*args, **kwargs)
        if self.args.save_total_limit is not None:
            raise ValueError('Online dev requires retaining all checkpoints')

    def create_scheduler(self, num_training_steps, optimizer=None):
        if self.lr_scheduler is None:
            self.lr_scheduler = WarmupPlateauScheduler(
                optimizer if optimizer is not None else self.optimizer,
                warmup_steps=self.feedback_warmup_steps)
            self._created_lr_scheduler = True
        return self.lr_scheduler

    def _feedback_scheduler(self):
        return getattr(self.lr_scheduler, 'scheduler', self.lr_scheduler)

    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix='eval'):
        started = time.monotonic()
        step = self.state.global_step
        before = [group['lr'] for group in self.optimizer.param_groups]
        result = self.dev_evaluator(self.model, step)
        official = result.get('official', result)
        if any(key not in official for key in OFFICIAL_KEYS):
            raise ValueError('Online dev requires all 18 official P/R/F1 metrics')
        official = {key: float(official[key]) for key in OFFICIAL_KEYS}
        if any(not math.isfinite(value) or not 0 <= value <= 1 for value in official.values()):
            raise ValueError('Official dev metrics must be finite fractions')
        actions = self.feedback_controller.update(step, official['AC_f1'])
        if actions['reduce_lr']:
            self._feedback_scheduler().reduce_lr()
        self._feedback_new_best = actions['new_best']
        if actions['new_best']:
            self.state.best_metric = official['AC_f1']
            self.state.best_global_step = step
            self.state.best_model_checkpoint = str(Path(self.args.output_dir)/f'checkpoint-{step}')
        metrics = {f'{metric_key_prefix}_{key}': value for key, value in official.items()}
        self.log(metrics)
        self.control = self.callback_handler.on_evaluate(self.args, self.state, self.control, metrics)
        # Force saving after callbacks; feedback stop must not be cleared by one.
        self.control.should_save = True
        self.control.should_training_stop = self.control.should_training_stop or actions['stop']
        record = {'step': step, 'epoch': self.state.epoch, 'official': official,
                  'LR_before': before,
                  'LR_after': [group['lr'] for group in self.optimizer.param_groups],
                  'lr_before': before[0], 'lr_after': self.optimizer.param_groups[0]['lr'],
                  'score': official['AC_f1'],
                  **self.feedback_controller.state_dict(), **actions, 'raw_best': self.feedback_controller.best_score,
                  'raw_best_step': self.feedback_controller.best_step,
                  'wall_seconds': time.monotonic()-started,
                  'paths': result.get('paths', {})}
        with self.feedback_path.open('a', encoding='utf-8') as stream:
            stream.write(json.dumps(record, allow_nan=False)+'\n')
        return metrics

    def _determine_best_metric(self, metrics, trial):
        # Controller owns raw improvements and exact later-step ties.
        return self._feedback_new_best

    def _save_checkpoint(self, model, trial):
        super()._save_checkpoint(model, trial)
        directory = Path(self._get_output_dir(trial))/f'checkpoint-{self.state.global_step}'
        _write_json(directory/'feedback_state.json', {
            'format': 'online-dev-feedback-v1',
            'controller': self.feedback_controller.state_dict(),
            'observed_microbatches': self.observed_microbatches})
        manifest = json.loads((directory/'continuity.json').read_text())
        manifest['sha256']['feedback_state.json'] = _hash(directory/'feedback_state.json')
        _write_json(directory/'continuity.json', manifest)

    def _load_from_checkpoint(self, resume_from_checkpoint, model=None):
        validate_online_checkpoint(resume_from_checkpoint, expected_horizon=self.args.max_steps)
        super()._load_from_checkpoint(resume_from_checkpoint, model)
        saved = json.loads((Path(resume_from_checkpoint)/'feedback_state.json').read_text())
        self.feedback_controller.load_state_dict(saved['controller'])
        if self.feedback_controller.state_dict() != saved['controller']:
            raise ValueError('Feedback state was not restored exactly')
        self.observed_microbatches = saved['observed_microbatches']
        self.resume_evidence['feedback_exact'] = True
