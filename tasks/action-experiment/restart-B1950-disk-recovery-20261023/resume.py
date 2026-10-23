"""Explicit B1950 recovery with adaptive dev memory and a 60-second GPU gate."""
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import runpy
import sys
import time
import traceback
import signal
import subprocess
from contextlib import contextmanager

HERE = Path(__file__).resolve().parent
ACTION = HERE.parent
V6 = ACTION / 'recovery_v6'
LAUNCH = ACTION / 'launch-stable-B1490-v6-20261021'
GATE = ACTION / 'pause-B1507-resume60-20261022/idle_resource_loader.py'


@contextmanager
def model_lease(path=None):
    with Path(path or ACTION / 'project-model.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield


@contextmanager
def ignore_stop_signals():
    previous = {sig: signal.signal(sig, signal.SIG_IGN) for sig in (signal.SIGINT, signal.SIGTERM)}
    try:
        yield
    finally:
        for sig, handler in previous.items():
            signal.signal(sig, handler)


class ManagedProcess:
    def __init__(self, process):
        self.process = process
        self.pid = process.pid

    def stop(self):
        if self.process.poll() is None:
            with ignore_stop_signals():
                self.process.terminate()
                self.process.wait()

    def wait(self):
        try:
            return self.process.wait()
        except BaseException:
            self.stop()
            raise


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1048576), b''):
            value.update(block)
    return value.hexdigest()


def verify_extension():
    locks = read(HERE / 'source-sha256.json')
    required = {str(HERE / name) for name in ('resume.py', 'safe_evaluation.py', 'request.json')}
    required.add(str(GATE))
    if not required.issubset(locks):
        raise ValueError('Incomplete evaluation/resource extension source lock')
    for name, expected in locks.items():
        if sha(name) != expected:
            raise ValueError('Extension source changed: ' + name)


def record_policy():
    request = read(HERE / 'request.json')
    policy = dict(updated_unix=time.time(), user_authorized_resume=True,
        checkpoint=request['checkpoint'], numerical_sources='unchanged frozen recovery_v6',
        source_lock=str(HERE / 'source-sha256.json'),
        source_lock_sha256=sha(HERE / 'source-sha256.json'),
        evaluation=dict(requested_max_batch_size=4, effective_max_batch_size=2, cuda_oom_fallback=[1],
            terminal_batch_one_oom=True, actual_batch_sizes_recorded=True,
            greedy=True, max_new_tokens=768),
        allocator_configuration=os.environ.get('PYTORCH_CUDA_ALLOC_CONF'),
        GPU_admission=dict(continuous_idle_seconds=60, minimum_free_mib=21000,
            no_foreign_compute=True, no_reservation=True),
        auto_restart=False)
    destination = Path(request['output']) / 'restart-step1950-policy.json'
    temporary = destination.with_suffix('.tmp')
    temporary.write_text(json.dumps(policy, indent=2) + '\n')
    os.replace(temporary, destination)


def trainer():
    with model_lease():
        assert_no_models()
        _run_trainer()


def _run_trainer():
    verify_extension()
    sys.path.insert(0, str(V6))
    sys.path.insert(0, str(ACTION))
    sys.path.insert(0, str(HERE))
    import action_evaluation
    from safe_evaluation import evaluate_model
    action_evaluation.evaluate_model = evaluate_model
    spec = importlib.util.spec_from_file_location('idle_resource_loader', GATE)
    gate = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(gate)
    sys.modules['idle_resource_loader'] = gate
    record_policy()
    sys.argv = [str(V6 / 'train_grpo_recoverable.py'), *sys.argv[2:]]
    print('B1950 explicit resume: dev effective cap2, CUDA OOM retry1; expandable allocator; continuous idle60; frozen numerical sources', flush=True)
    runpy.run_path(str(V6 / 'train_grpo_recoverable.py'), run_name='__main__')


def wrap_command(command):
    if Path(command[2]).resolve() != V6 / 'train_grpo_recoverable.py':
        raise ValueError('Unexpected trainer command')
    return [*command[:2], str(HERE / 'resume.py'), '--trainer', *command[3:]]


def is_project_model(args, cwd):
    if not args or not Path(args[0]).name.startswith('python'):
        return False
    scripts = [Path(arg) if Path(arg).is_absolute() else Path(cwd) / arg
               for arg in args[1:] if arg.endswith('.py')]
    for script in scripts:
        script = script.resolve()
        if script.is_relative_to(ACTION) and ('--trainer' in args or script.name in
                ('train_grpo_recoverable.py', 'verify_gpu.py', 'probe.py', 'probe_history.py',
                 'evaluate_saved_adapter.py')):
            return True
    return False


def assert_no_models():
    for process in Path('/proc').iterdir():
        if not process.name.isdigit() or int(process.name) == os.getpid():
            continue
        try:
            args = (process / 'cmdline').read_bytes().decode().split('\0')
            cwd = (process / 'cwd').resolve()
        except (OSError, UnicodeError):
            continue
        if is_project_model(args, cwd):
            raise RuntimeError('Another project model is active: ' + process.name)


def main():
    if 'TMUX' not in os.environ:
        raise RuntimeError('Run inside tmux')
    # This must precede every torch import, including in the trainer child.
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
    if sys.argv[1:] and sys.argv[1] == '--trainer':
        trainer()
        return
    with (HERE / 'once.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (HERE / 'started.json').exists():
            raise RuntimeError('Explicit recovery already consumed; no automatic retry')
        verify_extension()
        assert_no_models()
        sys.path.insert(0, str(LAUNCH))
        import run_controller as controller
        request = read(HERE / 'request.json')
        checkpoint = controller.verify_checkpoint(request['checkpoint'], controller.OUTPUT)
        if sha(checkpoint / 'manifest.json') != request['checkpoint_manifest_sha256']:
            raise ValueError('Explicit B1950 snapshot changed')
        if sha(Path(request['output']) / 'failure.json') != request['expected_failure_sha256']:
            raise ValueError('Failure changed since this recovery was authorized')
        original_command = controller.trainer_command
        controller.trainer_command = lambda *a, **kw: wrap_command(original_command(*a, **kw))
        (HERE / 'started.json').write_text(json.dumps(dict(started_unix=time.time(),
            checkpoint=str(checkpoint), controller_pid=os.getpid())) + '\n')
        child = None
        def start(command, log):
            nonlocal child
            child = ManagedProcess(subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT))
            return child
        def interrupted(signum, frame):
            raise KeyboardInterrupt('Controller interrupted by signal ' + str(signum))
        previous_handler = signal.signal(signal.SIGTERM, interrupted)
        try:
            controller.run_controller(resume_checkpoint=checkpoint, start=start)
        finally:
            if child is not None:
                child.stop()
            signal.signal(signal.SIGTERM, previous_handler)


if __name__ == '__main__':
    try:
        main()
    except BaseException as error:
        (HERE / 'failure.json').write_text(json.dumps(dict(error=repr(error),
            traceback=traceback.format_exc(), updated_unix=time.time(),
            auto_restart=False), indent=2) + '\n')
        raise
