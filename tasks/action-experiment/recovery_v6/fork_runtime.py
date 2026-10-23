"""Fork a trusted, committed recovery boundary without modifying its source run."""
import copy
import hashlib
import json
from pathlib import Path
import shutil


SOURCE_METADATA_KEYS = {'source_sha256', 'precision_transition', 'continuation_mode',
    'warm_start_run', 'warm_start_step', 'optimizer_reset_at_warm_start', 'rng_reset_at_warm_start'}


def digest(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


def _read(path):
    return json.loads(Path(path).read_text())


def _prefix_digest(path, size):
    result = hashlib.sha256()
    if size:
        with Path(path).open('rb') as stream:
            while size:
                block = stream.read(min(size, 1024 * 1024))
                if not block:
                    raise ValueError('Source log shorter than committed prefix')
                result.update(block)
                size -= len(block)
    return result.hexdigest()


def _copy_tree_verified(source, target, snapshot):
    shutil.copytree(source, target)
    expected = {str(Path(path).relative_to(source)): value for path, value in snapshot.items()
                if source in Path(path).parents}
    actual = {str(path.relative_to(target)) for path in target.rglob('*') if path.is_file()}
    if actual != set(expected):
        raise ValueError('Inherited artifact copy inventory mismatch: ' + str(target))
    for name, value in expected.items():
        if digest(target / name) != value:
            raise ValueError('Inherited artifact copy hash mismatch: ' + str(target / name))


def _validate_checkpoint_files(checkpoint):
    if (checkpoint / 'manifest.json').is_symlink():
        raise ValueError('Checkpoint manifest must not be a symlink')
    manifest = _read(checkpoint / 'manifest.json')
    if manifest.get('format_version') != 1:
        raise ValueError('Unsupported checkpoint format')
    for relative, expected in manifest['files'].items():
        path = checkpoint / relative
        if (Path(relative).is_absolute() or '..' in Path(relative).parts
                or any(item.is_symlink() for item in (path, *path.parents) if item != checkpoint.parent)
                or not path.is_file() or digest(path) != expected):
            raise ValueError('Invalid checkpoint file: ' + relative)
    actual = {str(p.relative_to(checkpoint)) for p in checkpoint.rglob('*')
              if p.is_file() and p != checkpoint / 'manifest.json'}
    if actual != set(manifest['files']):
        raise ValueError('Checkpoint inventory mismatch')
    return manifest


def validate_resume_source(output, checkpoint, configuration, init_adapter, verify_locks):
    """Reject mismatched domain or foreign state before any run file is written."""
    import torch
    output, checkpoint = Path(output).resolve(), Path(checkpoint).resolve()
    if checkpoint.parent != output / 'recovery-checkpoints':
        raise ValueError('Resume checkpoint must belong to this output run')
    run = _read(output / 'provenance.json')
    old = _read(output / 'configuration.json')
    if Path(run['init_adapter']).resolve() != Path(init_adapter).resolve():
        raise ValueError('Resume changes frozen SFT reference')
    for key in old.keys() | configuration.keys():
        if key not in SOURCE_METADATA_KEYS and old.get(key) != configuration.get(key):
            raise ValueError('Resume configuration differs: ' + key)
    verify_locks(run['input_sha256'])
    _validate_checkpoint_files(checkpoint)
    state = torch.load(checkpoint / 'training-state.pt', map_location='cpu', weights_only=False)
    payload = state['payload']
    if payload.get('arithmetic') != configuration.get('arithmetic'):
        raise ValueError('Resume checkpoint arithmetic differs from configuration')
    if (state.get('format_version') != 1 or payload.get('pending') != 0 or state.get('gradients')
            or payload['next_group'] != payload['step'] * configuration['gradient_accumulation']):
        raise ValueError('Resume requires a complete optimizer boundary without gradients')
    return run


def validate_resume_train_order(output, planned):
    if _read(Path(output) / 'planned-train-order.json') != planned:
        raise ValueError('Resume train order differs from persisted plan')


def validate_fork_source(source, checkpoint, output, configuration, init_adapter, verify_locks):
    """Validate all input files before a new output directory is created.

    This accepts only trusted local torch snapshots, like load_checkpoint itself.
    The checkpoint and every production artifact are opened read-only.
    """
    import torch
    source, checkpoint, output = map(lambda p: Path(p).resolve(), (source, checkpoint, output))
    if output.exists() or output == source or source in output.parents or output in source.parents:
        raise ValueError('Fork output must be a new directory outside the source run')
    if checkpoint.parent != source / 'recovery-checkpoints':
        raise ValueError('Fork checkpoint must belong to the source run')
    old = _read(source / 'configuration.json')
    provenance = _read(source / 'provenance.json')
    if Path(provenance['init_adapter']).resolve() != Path(init_adapter).resolve():
        raise ValueError('Fork changes frozen SFT reference')
    # Only source implementation/arithmetic and continuation metadata may change.
    excluded = SOURCE_METADATA_KEYS | {'arithmetic', 'evaluation_batch_size'}
    previous_batch = old.get('evaluation_batch_size', 4)
    evaluation_batch = configuration.get('evaluation_batch_size', 1)
    if (type(previous_batch) is not int or type(evaluation_batch) is not int
            or previous_batch not in (1, 2, 4) or evaluation_batch not in (1, 2, 4)):
        raise ValueError('Fork supports only verified evaluation batch choices 1, 2, 4')
    for key in old.keys() | configuration.keys():
        if key not in excluded and old.get(key) != configuration.get(key):
            raise ValueError('Fork changes training configuration: ' + key)
    verify_locks(provenance['input_sha256'])
    _validate_checkpoint_files(checkpoint)
    state = torch.load(checkpoint / 'training-state.pt', map_location='cpu', weights_only=False)
    payload = state['payload']
    if (state.get('format_version') != 1 or payload.get('pending') != 0 or state.get('gradients')
            or payload['next_group'] != payload['step'] * old['gradient_accumulation']):
        raise ValueError('Fork requires a complete optimizer boundary without gradients')
    for name, size in payload['log_offsets'].items():
        if (Path(name).name != name or not name.endswith('.jsonl') or not isinstance(size, int)
                or isinstance(size, bool) or size < 0):
            raise ValueError('Invalid committed log offset')
        path = source / name
        if size and (not path.is_file() or path.is_symlink() or path.stat().st_size < size):
            raise ValueError('Source log shorter than committed offset: ' + name)
        if size:
            with path.open('rb') as stream:
                stream.seek(size - 1)
                if stream.read(1) != b'\n':
                    raise ValueError('Committed offset does not end at a complete JSONL record')
    snapshot = {}
    last_dev = payload['early_stopping']['last_step']
    for parent in ('dev', 'checkpoints'):
        for directory in (source / parent).glob('step-*'):
            if int(directory.name[5:]) <= last_dev:
                if directory.is_symlink():
                    raise ValueError('Inherited artifacts must not contain symlinks')
                for path in directory.rglob('*'):
                    if path.is_symlink():
                        raise ValueError('Inherited artifacts must not contain symlinks')
                    if path.is_file():
                        snapshot[str(path)] = digest(path)
    planned = source / 'planned-train-order.json'
    if not planned.is_file() or planned.is_symlink():
        raise ValueError('Fork requires a regular persisted train order')
    snapshot[str(planned)] = digest(planned)
    if payload['best_step'] is not None:
        selected = source / 'checkpoints' / f"step-{payload['best_step']}" / 'policy'
        for filename in ('adapter_model.safetensors', 'adapter_config.json'):
            if str(selected / filename) not in snapshot:
                raise ValueError('Fork is missing its selected adapter: ' + filename)
    prefix_hashes = {name: _prefix_digest(source / name, size)
                     for name, size in payload['log_offsets'].items()}
    transition = dict(source_run=str(source), source_checkpoint=str(checkpoint),
        source_manifest_sha256=digest(checkpoint / 'manifest.json'), source_step=payload['step'],
        source_next_group=payload['next_group'], previous_arithmetic=old.get('arithmetic',
            {'model_dtype': 'bfloat16', 'attention': 'sdpa', 'logprob': 'fp32_after_model'}),
        arithmetic=copy.deepcopy(configuration['arithmetic']), optimizer_restored=True, rng_restored=True)
    transition.update(previous_evaluation_batch_size=previous_batch,
        evaluation_batch_size=evaluation_batch,
        evaluation_batch_reason='Selected by longest-dev GPU acceptance; data and generation settings unchanged',
        inherited_file_sha256=snapshot,inherited_log_prefix_sha256=prefix_hashes,
        inherited_hash_baseline='Captured at fork time; no pre-existing historical hash baseline was available')
    return dict(source=source, checkpoint=checkpoint, payload=payload, transition=transition,
                inherited_input_sha256=snapshot)


def copy_committed_prefix(fork, output):
    """Copy only committed artifacts into an already prepared *new* run."""
    source, output, payload = fork['source'], Path(output), fork['payload']
    for name, size in payload['log_offsets'].items():
        remaining = size
        with (output / name).open('xb') as target:
            if remaining:
                with (source / name).open('rb') as stream:
                    while remaining:
                        block = stream.read(min(1024 * 1024, remaining))
                        if not block:
                            raise ValueError('Source log changed while copying: ' + name)
                        target.write(block)
                        remaining -= len(block)
        if digest(output / name) != fork['transition']['inherited_log_prefix_sha256'][name]:
            raise ValueError('Inherited log copy hash mismatch: ' + name)
    # Copy dev checkpoints, rather than retain writable links into the old run.
    last_dev = payload['early_stopping']['last_step']
    for parent in ('dev', 'checkpoints'):
        for path in (source / parent).glob('step-*'):
            if int(path.name[5:]) <= last_dev:
                _copy_tree_verified(path, output / parent / path.name, fork['inherited_input_sha256'])
    # Keep the exact log offsets for load_checkpoint/truncate_logs. Adapter paths
    # in inherited history continue to name the immutable source; local copied
    # checkpoints are used by reconcile_artifacts when reconstructing best.
    best = payload['best_step']
    if best is not None:
        selected = output / 'checkpoints' / f'step-{best}' / 'policy'
        if not selected.is_dir():
            raise ValueError('Fork is missing its selected committed adapter')
        source_selected = source / 'checkpoints' / f'step-{best}' / 'policy'
        _copy_tree_verified(source_selected, output / 'best-adapter', fork['inherited_input_sha256'])
    planned = source / 'planned-train-order.json'
    if planned.is_file():
        shutil.copy2(planned, output / planned.name)
        if digest(output / planned.name) != fork['inherited_input_sha256'][str(planned)]:
            raise ValueError('Inherited train-order copy hash mismatch')
    return copy.deepcopy(fork['transition'])


def validate_fork_train_order(fork, planned):
    """Do not silently reuse optimizer/cursor state for a reordered dataset."""
    original = _read(fork['source'] / 'planned-train-order.json')
    if original['ids'] != planned['ids']:
        raise ValueError('Fork train order differs from source run')
