"""Explicit one-stage B precision-transition queue; never retry failed training."""
import argparse
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import time

LAUNCH = Path(__file__).resolve().parent
ACTION = LAUNCH.parent
V6 = ACTION / 'recovery_v6'
PYTHON = '/home/myluo/miniconda3/envs/qwen3meld-run/bin/python'
OLD_B = ACTION / 'runs/r0-v2-B-recoverable-v5-42-20261019'
SOURCE_CHECKPOINT = OLD_B / 'recovery-checkpoints/exception-step1490-1792592102711572228'
OUTPUT = ACTION / 'runs/r0-v2-B-stable-v6-from1490-42-20261021'
SFT = ACTION / 'runs/action-sft-v2-42-20261012/best-adapter'
ACCEPTANCE = ACTION / 'verification-stable-v6-20261021'


def read(path):
    return json.loads(Path(path).read_text())


def digest(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def write_state(launch, **state):
    path = Path(launch) / 'pipeline-state.json'
    temporary = path.with_suffix('.tmp')
    with temporary.open('w') as stream:
        json.dump(dict(**state, updated_unix=time.time()), stream, ensure_ascii=False, indent=2)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)
    with (Path(launch) / 'pipeline.log').open('a') as log:
        log.write(json.dumps(dict(updated_unix=time.time(), status=state.get('status'),
            stage=state.get('stage'), model_pid=state.get('model_pid'),
            error=state.get('error')), ensure_ascii=False) + '\n')


def verify_sources(launch=LAUNCH):
    locks = read(Path(launch) / 'source-sha256.json')
    required = {str(Path(__file__).resolve()), *[str(path.resolve()) for path in V6.glob('*.py')]}
    required.update(str((ACCEPTANCE / name).resolve()) for name in
        ('result.json', 'gpu.exit', 'runtime-source-sha256.json'))
    if not required.issubset(locks):
        raise ValueError('Deployment source lock is incomplete; do not run an unfrozen draft')
    for filename, expected in locks.items():
        if digest(filename) != expected:
            raise ValueError('Deployment source changed: ' + filename)
    return verify_acceptance()


def _finite_number(value):
    return type(value) in (int, float) and math.isfinite(value)


def _positive_integer(value):
    return type(value) is int and value > 0


def _verify_training_evidence(result):
    stages = result.get('stages')
    if not isinstance(stages, dict) or result.get('production_runs_modified') is not False:
        raise ValueError('GPU acceptance lacks unchanged production evidence')
    forwards = stages.get('real_forward')
    if not isinstance(forwards, list) or len(forwards) != 6:
        raise ValueError('GPU acceptance requires six real forward responses')
    identities = set()
    for row in forwards:
        if (not isinstance(row, dict) or row.get('run') not in ('A', 'B')
                or type(row.get('group')) is not int or type(row.get('sample')) is not int
                or not _positive_integer(row.get('tokens'))
                or not _finite_number(row.get('max_difference'))
                or not 0 <= row['max_difference'] < .001):
            raise ValueError('GPU acceptance real forward probability evidence is invalid')
        identities.add((row['run'], row['group'], row['sample']))
    if len(identities) != 6 or any(sum(row['run'] == name for row in forwards) != 3 for name in ('A', 'B')):
        raise ValueError('GPU acceptance requires three distinct forward responses from each run')
    optimizer = stages.get('optimizer')
    if (not isinstance(optimizer, dict) or optimizer.get('initial_step') != 1490
            or optimizer.get('final_step') != 1491 or optimizer.get('policy_changed') is not True
            or optimizer.get('reference_unchanged') is not True
            or not _finite_number(optimizer.get('gradient_norm')) or optimizer['gradient_norm'] < 0):
        raise ValueError('GPU acceptance optimizer/gradient evidence is invalid')
    samples = optimizer.get('samples')
    if not isinstance(samples, list) or len(samples) != 16:
        raise ValueError('GPU acceptance requires sixteen optimizer samples')
    pairs = set()
    for row in samples:
        if (not isinstance(row, dict) or type(row.get('group')) is not int
                or type(row.get('sample')) is not int or not _positive_integer(row.get('tokens'))
                or not _finite_number(row.get('loss')) or not _finite_number(row.get('max_difference'))
                or not 0 <= row['max_difference'] <= 1.0):
            raise ValueError('GPU acceptance optimizer sample evidence is invalid')
        pairs.add((row['group'], row['sample']))
    if pairs != {(group, sample) for group in range(5960, 5964) for sample in range(4)}:
        raise ValueError('GPU acceptance optimizer samples do not cover the next complete update')
    point = stages.get('checkpoint')
    if (not isinstance(point, dict) or point.get('step') != 1491
            or point.get('rng_equal') is not True or point.get('policy_equal') is not True):
        raise ValueError('GPU acceptance checkpoint restoration evidence is invalid')
    train = stages.get('longest_train')
    if (not isinstance(train, dict) or not _positive_integer(train.get('prompt_tokens'))
            or train.get('completion_tokens') != 768
            or train.get('total_tokens') != train['prompt_tokens'] + 768
            or train.get('finite_gradients') is not True
            or not _positive_integer(train.get('peak_allocated_bytes'))):
        raise ValueError('GPU acceptance longest-train backward evidence is invalid')
    return stages


def verify_acceptance(acceptance=ACCEPTANCE, *, required_runtime=None):
    """Validate the completed, matching GPU acceptance before freezing/deploying."""
    acceptance = Path(acceptance)
    if (acceptance / 'gpu.exit').read_text().strip() != '0':
        raise ValueError('GPU acceptance did not exit successfully')
    result = read(acceptance / 'result.json')
    if result.get('status') != 'completed':
        raise ValueError('GPU acceptance is not completed')
    stages = _verify_training_evidence(result)
    longest = stages.get('longest_dev')
    if not isinstance(longest, dict):
        raise ValueError('GPU acceptance lacks longest-dev evidence')
    batch = longest['batch_size']
    if (type(batch) is not int or batch not in (1, 2, 4) or longest.get('forced_generated_tokens') != 768
            or not _positive_integer(longest.get('prompt_tokens'))
            or not _positive_integer(longest.get('peak_allocated_bytes'))):
        raise ValueError('GPU acceptance lacks the full longest-dev batch validation')
    if not any(isinstance(item, dict) and item.get('batch_size') == batch and item.get('status') == 'passed'
               and _finite_number(item.get('wall_seconds')) and item['wall_seconds'] > 0
               and _positive_integer(item.get('peak_allocated_bytes'))
               for item in stages.get('dev_batch_benchmark', [])):
        raise ValueError('Selected evaluation batch has no passed GPU benchmark')
    runtime = read(acceptance / 'runtime-source-sha256.json')
    if required_runtime is None:
        required_runtime = {str((ACTION / name).resolve()) for name in
            ('recovery_v6/stable_arithmetic.py', 'grpo_core.py', 'action_common.py', 'action_rewards.py',
             'action_reward_v2.py', 'action_update_mask.py', 'grpo_diagnostics.py', 'recovery_v6/verify_gpu.py')}
    if not set(required_runtime).issubset(runtime):
        raise ValueError('Acceptance numerical runtime source lock is incomplete')
    for filename, expected in runtime.items():
        if digest(filename) != expected:
            raise ValueError('Acceptance runtime source changed: ' + filename)
    return batch


def verify_checkpoint(checkpoint, output):
    checkpoint, output = Path(checkpoint).resolve(), Path(output).resolve()
    if checkpoint.parent != output / 'recovery-checkpoints':
        raise ValueError('Resume requires a checkpoint inside the new B run')
    manifest_path = checkpoint / 'manifest.json'
    if manifest_path.is_symlink():
        raise ValueError('Checkpoint manifest cannot be a symlink')
    manifest = read(manifest_path)
    if manifest.get('format_version') != 1:
        raise ValueError('Unsupported recovery format')
    for name, expected in manifest['files'].items():
        path = checkpoint / name
        if (Path(name).is_absolute() or '..' in Path(name).parts or path.is_symlink()
                or not path.is_file() or digest(path) != expected):
            raise ValueError('Incomplete recovery checkpoint: ' + name)
    actual = {str(path.relative_to(checkpoint)) for path in checkpoint.rglob('*')
              if path.is_file() and path != manifest_path}
    if actual != set(manifest['files']):
        raise ValueError('Recovery checkpoint inventory mismatch')
    return checkpoint


def trainer_command(output=OUTPUT, resume_checkpoint=None, evaluation_batch_size=1):
    command = [PYTHON, '-u', str(V6 / 'train_grpo_recoverable.py'), '--output', str(output),
        '--init-adapter', str(SFT), '--reward-version', 'r0_v2', '--policy-update-mask', 'field_v2',
        '--gradient-accumulation', '4', '--dev-steps', '400', '--max-steps', '3207',
        '--early-stopping-patience', '3', '--checkpoint-steps', '25', '--gpu-target-mib', '21000',
        '--evaluation-batch-size', str(evaluation_batch_size)]
    if resume_checkpoint is None:
        command += ['--fork-from-run', str(OLD_B), '--resume-checkpoint', str(SOURCE_CHECKPOINT)]
    else:
        command += ['--resume-checkpoint', str(resume_checkpoint)]
    return command


def completed(output):
    try:
        output = Path(output)
        metrics = read(output / 'train_metrics.json')
        valid = (metrics.get('status') == 'completed' and not metrics.get('smoke')
            and metrics.get('policy_update_mask_mode') == 'field_v2'
            and metrics.get('global_step', 0) >= 1490
            and metrics.get('arithmetic', {}).get('protocol') == 'fp32-body-bf16-storage-v1')
        if not valid:
            return False
        pointer = read(output / 'recovery-checkpoints/latest.json')
        name = pointer['checkpoint']
        if Path(name).name != name or not name.startswith(f"complete-step{metrics['global_step']}-"):
            return False
        point = output / 'recovery-checkpoints' / name
        if digest(point / 'manifest.json') != pointer['manifest_sha256']:
            return False
        verify_checkpoint(point, output)
        return (digest(output / 'best-adapter/adapter_model.safetensors') == metrics['best_adapter_sha256']
            and digest(output / 'best-adapter/adapter_config.json') == metrics['best_adapter_config_sha256'])
    except (OSError, ValueError, TypeError, KeyError):
        return False


def start_model(command, log):
    return subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)


def run_controller(launch=LAUNCH, output=OUTPUT, resume_checkpoint=None,
                   *, verify=verify_sources, start=start_model, evaluation_batch_size=None):
    """An explicit resume flag is the only way to reuse an existing run."""
    launch, output = Path(launch), Path(output)
    with (launch / 'pipeline.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        accepted_batch = verify(launch)
        if (evaluation_batch_size is not None and accepted_batch is not None
                and evaluation_batch_size != accepted_batch):
            raise ValueError('Controller batch differs from completed GPU acceptance')
        if evaluation_batch_size is None:
            evaluation_batch_size = accepted_batch
        if type(evaluation_batch_size) is not int or evaluation_batch_size not in (1, 2, 4):
            raise ValueError('Controller requires a GPU-verified evaluation batch')
        state_path = launch / 'pipeline-state.json'
        previous = read(state_path) if state_path.exists() else {}
        if previous.get('status') == 'completed':
            if completed(output):
                return
            raise ValueError('Completed queue has missing or inconsistent output')
        if resume_checkpoint is None:
            if output.exists() or previous.get('status') in {'failed', 'running'}:
                raise ValueError('Existing/failed B requires explicit --resume-checkpoint; no automatic retry')
        else:
            resume_checkpoint = verify_checkpoint(resume_checkpoint, output)
        command = trainer_command(output, resume_checkpoint, evaluation_batch_size)
        state = dict(stage='B', output=str(output.resolve()), command=command,
            source_run=str(OLD_B), source_checkpoint=str(SOURCE_CHECKPOINT),
            controller_pid=os.getpid(), stages=['B'], auto_restart=False,
            explicit_resume=resume_checkpoint is not None, started_unix=time.time())
        state['evaluation_batch_size'] = evaluation_batch_size
        write_state(launch, status='running', **state)
        try:
            with (launch / 'B.log').open('a') as log:
                process = start(command, log)
                state['model_pid'] = process.pid
                write_state(launch, status='running', **state)
                code = process.wait()
            if code:
                raise RuntimeError('B trainer exited with code ' + str(code))
            if not completed(output):
                raise RuntimeError('B exited without verified completion metrics')
        except BaseException as error:
            write_state(launch, status='failed', error=repr(error), later_stages_started=False, **state)
            raise
        write_state(launch, status='completed', **state)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--resume-checkpoint', type=Path,
        help='Explicitly resume a complete checkpoint in the NEW B run; never select latest automatically')
    args = parser.parse_args()
    if 'TMUX' not in os.environ:
        raise RuntimeError('Run the controller inside tmux')
    run_controller(resume_checkpoint=args.resume_checkpoint)


if __name__ == '__main__':
    main()
