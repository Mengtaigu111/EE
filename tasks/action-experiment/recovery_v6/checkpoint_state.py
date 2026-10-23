"""Trusted local GRPO recovery snapshots, with verified atomic publication.

The caller exports/imports its policy adapter through callbacks. A checkpoint is
published only after every file has been written and hashed; latest.json changes
last. Only load trusted checkpoints: torch state includes Python pickle objects.
"""
from __future__ import annotations

import copy
import hashlib
import json
import os
import random
import shutil
import tempfile
from pathlib import Path
from typing import Any, Callable

import torch

FORMAT_VERSION = 1


def _cpu_copy(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: _cpu_copy(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_cpu_copy(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_cpu_copy(item) for item in value)
    return copy.deepcopy(value)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, value: dict):
    with path.open('w', encoding='utf-8') as stream:
        json.dump(value, stream, sort_keys=True, indent=2)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())


def _fsync_directory(path: Path):
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _signature(policy):
    return [(name, tuple(parameter.shape), str(parameter.dtype), parameter.requires_grad)
            for name, parameter in policy.named_parameters()]


def capture_rng_state(capture_cuda: bool = True) -> dict:
    """Capture RNG cheaply at a rollback boundary without copying optimizer."""
    return {'python': random.getstate(), 'torch_cpu': torch.get_rng_state(),
            'torch_cuda': torch.cuda.get_rng_state_all()
            if capture_cuda and torch.cuda.is_initialized() else None}


def restore_rng_state(state: dict):
    random.setstate(state['python'])
    torch.set_rng_state(state['torch_cpu'])
    cuda_states = state['torch_cuda']
    if cuda_states is not None:
        if not torch.cuda.is_available() or len(cuda_states) != torch.cuda.device_count():
            raise ValueError('CUDA RNG device topology mismatch')
        torch.cuda.set_rng_state_all(cuda_states)


def resolve_latest(root: str | Path) -> Path:
    root = Path(root)
    pointer = json.loads((root / 'latest.json').read_text())
    name = pointer['checkpoint']
    if not isinstance(name, str) or Path(name).name != name or name in {'.', '..'}:
        raise ValueError('Invalid latest checkpoint name')
    target = root / name
    if _sha256(target / 'manifest.json') != pointer['manifest_sha256']:
        raise ValueError('Latest manifest hash mismatch')
    return target


def save_checkpoint(root: str | Path, name: str, policy, optimizer,
                    payload: dict[str, Any], export_policy: Callable | None = None) -> Path:
    """Export adapter and full resumable state, then atomically publish latest.

    export_policy(policy, adapter_directory) must create that directory itself.
    payload contains caller loop/early-stop/probability-guard state. It may contain
    tensors; tensors are detached and copied to CPU without modifying live state.
    Never overwrite an existing named checkpoint.
    """
    if not name or Path(name).name != name or name in {'.', '..', 'latest.json'}:
        raise ValueError('Checkpoint name must be a single directory name')
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    final = root / name
    if final.exists():
        raise FileExistsError(final)
    temporary = Path(tempfile.mkdtemp(prefix=f'.{name}.tmp-', dir=root))
    pointer_temporary = None
    rng_state = capture_rng_state()
    try:
        adapter = temporary / 'adapter'
        if export_policy is None:
            adapter.mkdir()
            torch.save(_cpu_copy(policy.state_dict()), adapter / 'model.pt')
        else:
            export_policy(policy, adapter)
        if not adapter.is_dir() or not any(adapter.rglob('*')):
            raise ValueError('Policy export produced no adapter files')
        state = {
            'format_version': FORMAT_VERSION,
            'parameter_signature': _signature(policy),
            'optimizer': _cpu_copy(optimizer.state_dict()),
            'gradients': {key: parameter.grad.detach().cpu().clone()
                          for key, parameter in policy.named_parameters()
                          if parameter.grad is not None},
            'rng': rng_state,
            'payload': _cpu_copy(payload),
        }
        torch.save(state, temporary / 'training-state.pt')
        paths = sorted(path for path in temporary.rglob('*') if path.is_file())
        for path in paths:
            if path.is_symlink():
                raise ValueError('Checkpoint files must not be symlinks')
            with path.open('rb') as stream:
                os.fsync(stream.fileno())
        manifest = {'format_version': FORMAT_VERSION,
                    'files': {str(path.relative_to(temporary)): _sha256(path) for path in paths}}
        _write_json(temporary / 'manifest.json', manifest)
        _fsync_directory(adapter)
        _fsync_directory(temporary)
        os.rename(temporary, final)
        _fsync_directory(root)
        descriptor, pointer_name = tempfile.mkstemp(prefix='.latest-', dir=root)
        os.close(descriptor)
        pointer_temporary = Path(pointer_name)
        _write_json(pointer_temporary, {'checkpoint': name,
                                      'manifest_sha256': _sha256(final / 'manifest.json')})
        os.replace(pointer_temporary, root / 'latest.json')
        _fsync_directory(root)
        return final
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
        if pointer_temporary is not None and pointer_temporary.exists():
            pointer_temporary.unlink()


def load_checkpoint(path: str | Path, policy, optimizer,
                    load_policy: Callable | None = None, restore_rng: bool = True) -> dict[str, Any]:
    """Verify all contents before restoring a trusted checkpoint.

    load_policy(policy, adapter_directory) imports adapter weights into the exact
    same parameter topology. optimizer must already reference policy parameters.
    """
    path = Path(path)
    manifest = json.loads((path / 'manifest.json').read_text())
    if manifest.get('format_version') != FORMAT_VERSION:
        raise ValueError('Unsupported checkpoint format')
    files = manifest['files']
    for relative, expected in files.items():
        target = path / relative
        if Path(relative).is_absolute() or '..' in Path(relative).parts or target.is_symlink():
            raise ValueError('Invalid checkpoint file path')
        if not target.is_file() or _sha256(target) != expected:
            raise ValueError(f'Checkpoint file hash mismatch: {relative}')
    actual = {str(item.relative_to(path)) for item in path.rglob('*')
              if item.is_file() and item != path / 'manifest.json'}
    if actual != set(files):
        raise ValueError('Checkpoint file inventory mismatch')
    # Explicitly trusted local state: payload includes Python RNG and guard state.
    state = torch.load(path / 'training-state.pt', map_location='cpu', weights_only=False)
    if state.get('format_version') != FORMAT_VERSION:
        raise ValueError('Unsupported training-state format')
    if state['parameter_signature'] != _signature(policy):
        raise ValueError('Policy parameter signature mismatch')
    if load_policy is None:
        policy.load_state_dict(torch.load(path / 'adapter' / 'model.pt', map_location='cpu', weights_only=True))
    else:
        load_policy(policy, path / 'adapter')
    optimizer.load_state_dict(state['optimizer'])
    parameters = dict(policy.named_parameters())
    for key in state['gradients']:
        if key not in parameters:
            raise ValueError(f'Unknown gradient parameter: {key}')
    for key, parameter in parameters.items():
        gradient = state['gradients'].get(key)
        parameter.grad = None if gradient is None else gradient.to(device=parameter.device, dtype=parameter.dtype)
    if restore_rng:
        restore_rng_state(state['rng'])
    return state['payload']
