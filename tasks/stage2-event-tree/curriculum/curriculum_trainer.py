"""Continuous curriculum checkpoints without pickle or a second LR schedule.

Transformers 4.52 rejects optimizer pickle resumes with baseline Torch 2.4.
These single-process Trainer hooks use JSON and safetensors instead, keeping
the baseline runtime and all optimizer, scheduler and random state intact.
"""
import hashlib
import json
from pathlib import Path
import random

import numpy as np
import torch
from safetensors.torch import load_file, save_file
from transformers import Trainer, TrainerCallback


def _write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def _pack(value, tensors):
    if isinstance(value, torch.Tensor):
        key = f'tensor_{len(tensors)}'
        tensors[key] = value.detach().cpu().contiguous().clone()
        return {'tensor': key}
    if isinstance(value, dict):
        return {'dict': [[_pack(k, tensors), _pack(v, tensors)] for k, v in value.items()]}
    if isinstance(value, (tuple, list)):
        return {type(value).__name__: [_pack(v, tensors) for v in value]}
    if value is None or isinstance(value, (bool, int, float, str)):
        return {'scalar': value}
    raise TypeError(f'Unsupported checkpoint value: {type(value).__name__}')


def _unpack(value, tensors):
    if 'tensor' in value:
        return tensors[value['tensor']]
    if 'dict' in value:
        return {_unpack(k, tensors): _unpack(v, tensors) for k, v in value['dict']}
    if 'tuple' in value:
        return tuple(_unpack(v, tensors) for v in value['tuple'])
    if 'list' in value:
        return [_unpack(v, tensors) for v in value['list']]
    return value['scalar']


def _save_state(directory, name, value):
    tensors = {}
    packed = _pack(value, tensors)
    save_file(tensors, str(directory / f'{name}.safetensors'))
    _write_json(directory / f'{name}.json', packed)


def _load_state(directory, name):
    return _unpack(json.loads((directory / f'{name}.json').read_text()),
                   load_file(str(directory / f'{name}.safetensors'), device='cpu'))


def _equal(left, right):
    if isinstance(left, torch.Tensor):
        return isinstance(right, torch.Tensor) and torch.equal(left.cpu(), right.cpu())
    if isinstance(left, dict):
        return isinstance(right, dict) and left.keys() == right.keys() and all(
            _equal(left[k], right[k]) for k in left)
    if isinstance(left, (tuple, list)):
        return type(left) is type(right) and len(left) == len(right) and all(
            _equal(a, b) for a, b in zip(left, right))
    return left == right


def _rng_state():
    numpy_state = np.random.get_state()
    return {'python': random.getstate(),
            'numpy': (numpy_state[0], torch.from_numpy(numpy_state[1].astype(np.int64)),
                      *numpy_state[2:]),
            'cpu': torch.random.get_rng_state(),
            'cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}


def _hash(path):
    result = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            result.update(chunk)
    return result.hexdigest()


def validate_resume_checkpoint(path, expected_step=None, expected_horizon=None):
    """Fail closed for missing, corrupt, or wrong-phase full checkpoints."""
    path = Path(path)
    state = json.loads((path / 'trainer_state.json').read_text())
    if expected_step is not None and state['global_step'] != expected_step:
        raise ValueError('Resume checkpoint has unexpected global step')
    if expected_horizon is not None and state['max_steps'] != expected_horizon:
        raise ValueError('Resume checkpoint has unexpected scheduler horizon')
    for filename in ('adapter_model.safetensors', 'adapter_config.json'):
        if not (path / filename).is_file():
            raise FileNotFoundError(path / filename)
    manifest = json.loads((path / 'continuity.json').read_text())
    if manifest['format'] != 'curriculum-safe-v1':
        raise ValueError('Unsupported curriculum checkpoint format')
    for filename, expected_hash in manifest['sha256'].items():
        if _hash(path / filename) != expected_hash:
            raise ValueError(f'Checkpoint state hash mismatch: {filename}')
    required = ['adapter_model.safetensors', 'adapter_config.json', 'trainer_state.json']
    required.extend(f'{name}.{extension}' for name in ('optimizer', 'scheduler', 'rng')
                    for extension in ('json', 'safetensors'))
    for filename in required:
        if filename not in manifest['sha256']:
            raise ValueError(f'Checkpoint manifest missing {filename}')
    return state


class PhaseStopCallback(TrainerCallback):
    """Stop and save at an optimizer boundary while retaining total max_steps."""
    def __init__(self, stop_step):
        if stop_step < 1:
            raise ValueError('Phase stop step must be positive')
        self.stop_step = stop_step

    def on_train_begin(self, args, state, control, **kwargs):
        if not state.global_step < self.stop_step <= args.max_steps:
            raise ValueError('Phase boundary must follow resume step and fit global horizon')

    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step >= self.stop_step:
            control.should_training_stop = True
            control.should_save = True
        return control


class _ResumeEntryCallback(TrainerCallback):
    def __init__(self, trainer):
        self.trainer = trainer

    def on_train_begin(self, args, state, control, **kwargs):
        evidence = self.trainer.resume_evidence
        if evidence:
            if state.global_step != evidence['expected_step']:
                raise ValueError('Trainer global step was not restored')
            if self.trainer.lr_scheduler.state_dict()['last_epoch'] != state.global_step:
                raise ValueError('Scheduler step does not match restored global step')
            evidence['global_step'] = state.global_step
            evidence['horizon'] = args.max_steps
            evidence['data_skip'] = not args.ignore_data_skip


class CurriculumTrainer(Trainer):
    """Single-process Trainer with safe, verified full-state checkpoint hooks."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.args.world_size != 1 or self.is_deepspeed_enabled or self.is_fsdp_enabled:
            raise ValueError('Curriculum checkpoint hooks require one ordinary process')
        self.resume_evidence = {}
        self.add_callback(_ResumeEntryCallback(self))

    def training_step(self, model, inputs, num_items_in_batch=None):
        loss = super().training_step(model, inputs, num_items_in_batch)
        if not torch.isfinite(loss).all().item():
            raise FloatingPointError(
                f'Nonfinite training loss before optimizer step {self.state.global_step + 1}')
        return loss

    def _load_from_checkpoint(self, resume_from_checkpoint, model=None):
        from peft import get_peft_model_state_dict
        state = validate_resume_checkpoint(resume_from_checkpoint,
                                           expected_horizon=self.args.max_steps)
        super()._load_from_checkpoint(resume_from_checkpoint, model)
        actual = get_peft_model_state_dict(model or self.model)
        expected = load_file(str(Path(resume_from_checkpoint) / 'adapter_model.safetensors'))
        if not _equal(expected, actual):
            raise ValueError('Adapter weights were not restored exactly')
        self.resume_evidence.update({'checkpoint': str(Path(resume_from_checkpoint).resolve()),
                                     'expected_step': state['global_step'], 'model_exact': True})

    def _save_optimizer_and_scheduler(self, output_dir):
        directory = Path(output_dir)
        _save_state(directory, 'optimizer', self.optimizer.state_dict())
        _save_state(directory, 'scheduler', self.lr_scheduler.state_dict())

    def _load_optimizer_and_scheduler(self, checkpoint):
        if checkpoint is None:
            return
        directory = Path(checkpoint)
        optimizer = _load_state(directory, 'optimizer')
        scheduler = _load_state(directory, 'scheduler')
        self.optimizer.load_state_dict(optimizer)
        self.lr_scheduler.load_state_dict(scheduler)
        if not _equal(optimizer, self.optimizer.state_dict()):
            raise ValueError('Optimizer state was not restored exactly')
        if not _equal(scheduler, self.lr_scheduler.state_dict()):
            raise ValueError('Scheduler state was not restored exactly')
        self.resume_evidence.update({'optimizer_exact': True, 'scheduler_exact': True,
            'scheduler_last_epoch': scheduler['last_epoch'],
            'learning_rates': [group['lr'] for group in self.optimizer.param_groups]})

    def _save_scaler(self, output_dir):
        if self.accelerator.scaler is not None:
            _save_state(Path(output_dir), 'scaler', self.accelerator.scaler.state_dict())

    def _load_scaler(self, checkpoint):
        if checkpoint is not None and self.accelerator.scaler is not None:
            state = _load_state(Path(checkpoint), 'scaler')
            self.accelerator.scaler.load_state_dict(state)
            if not _equal(state, self.accelerator.scaler.state_dict()):
                raise ValueError('Scaler state was not restored exactly')
            self.resume_evidence['scaler'] = 'restored exactly'
        elif checkpoint is not None:
            self.resume_evidence['scaler'] = 'not used (bf16 / CPU)'

    def _save_rng_state(self, output_dir):
        directory = Path(output_dir)
        _save_state(directory, 'rng', _rng_state())

    def _save_checkpoint(self, model, trial):
        # Native Trainer writes trainer_state.json after all state hooks finish.
        # Seal integrity only when the complete checkpoint is on disk.
        super()._save_checkpoint(model, trial)
        directory = Path(self._get_output_dir(trial)) / f'checkpoint-{self.state.global_step}'
        names = [f'{name}.{extension}' for name in ('optimizer', 'scheduler', 'rng')
                 for extension in ('json', 'safetensors')]
        names.extend(['adapter_model.safetensors', 'adapter_config.json', 'trainer_state.json'])
        if self.accelerator.scaler is not None:
            names.extend(['scaler.json', 'scaler.safetensors'])
        _write_json(directory / 'continuity.json', {'format': 'curriculum-safe-v1',
                    'sha256': {name: _hash(directory / name) for name in names}})

    def _load_rng_state(self, checkpoint):
        if checkpoint is None:
            return
        state = _load_state(Path(checkpoint), 'rng')
        random.setstate(state['python'])
        numpy_state = state['numpy']
        np.random.set_state((numpy_state[0], numpy_state[1].numpy().astype(np.uint32),
                             *numpy_state[2:]))
        torch.random.set_rng_state(state['cpu'])
        if state['cuda']:
            if not torch.cuda.is_available() or len(state['cuda']) != torch.cuda.device_count():
                raise ValueError('CUDA RNG state does not match current device count')
            torch.cuda.set_rng_state_all(state['cuda'])
        if not _equal(state, _rng_state()):
            raise ValueError('Random state was not restored exactly')
        self.resume_evidence['rng_exact'] = True
        Path(self.args.output_dir).mkdir(parents=True, exist_ok=True)
        _write_json(Path(self.args.output_dir) / 'resume_evidence.json', self.resume_evidence)
