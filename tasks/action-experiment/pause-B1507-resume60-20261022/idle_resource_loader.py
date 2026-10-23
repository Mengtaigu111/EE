"""Wait for an idle GPU without allocating or reserving any device memory."""
import math
import os
import subprocess
import time


def _physical_device(device):
    """Resolve a logical CUDA index to nvidia-smi's physical index or UUID."""
    if not isinstance(device, int) or device < 0:
        raise ValueError('device must be a nonnegative CUDA index')
    visible = os.environ.get('CUDA_VISIBLE_DEVICES')
    if visible is None:
        return str(device)
    devices = [item.strip() for item in visible.split(',')]
    if device >= len(devices) or not devices[device] or devices[device] == '-1':
        raise ValueError('requested device is not visible in CUDA_VISIBLE_DEVICES')
    return devices[device]


def query_gpu_state(device=0):
    """Read free MiB and all foreign compute PIDs; malformed output is an error."""
    physical = _physical_device(device)
    def read(field):
        result = subprocess.run(
            ['nvidia-smi', '--id=' + physical, field, '--format=csv,noheader,nounits'],
            capture_output=True, text=True, check=True, timeout=5)
        return result.stdout.strip()
    memory = read('--query-gpu=memory.free')
    if not memory.isdecimal():
        raise ValueError('invalid GPU free-memory response: ' + repr(memory))
    pids_text = read('--query-compute-apps=pid')
    pids = []
    for row in pids_text.splitlines():
        pid_text = row.strip()
        if not pid_text.isdecimal() or int(pid_text) <= 0:
            raise ValueError('invalid GPU compute PID response: ' + repr(row))
        pid = int(pid_text)
        if pid != os.getpid():
            pids.append(pid)
    return {'free_mib': int(memory), 'foreign_pids': sorted(set(pids))}


def _validate_snapshot(snapshot):
    free = snapshot['free_mib']
    pids = snapshot['foreign_pids']
    if type(free) is not int or free < 0:
        raise ValueError('free_mib must be a nonnegative integer')
    if not isinstance(pids, (list, tuple)) or any(type(pid) is not int or pid <= 0 for pid in pids):
        raise ValueError('foreign_pids must contain positive integer PIDs')
    return {'free_mib': free, 'foreign_pids': list(pids)}


def wait_for_idle_training_gpu(move_to_gpu, rollback_to_cpu, *, target_mib=21000,
        poll_seconds=1.0, on_state=None, query=None, oom_error=None,
        empty_cache=None, sleep=time.sleep, device=0,
        stable_seconds=60, clock=time.monotonic):
    """Load after continuously idle GPU admission; retry only CUDA OOMs.

    Waiting uses nvidia-smi only. In particular it creates no CUDA context,
    reservation buffers, or background thread. Every transfer attempt requires
    a new complete stable_seconds window; zero preserves immediate admission.
    """
    if target_mib <= 0 or poll_seconds <= 0:
        raise ValueError('target_mib and poll_seconds must be positive')
    if not math.isfinite(stable_seconds) or stable_seconds < 0:
        raise ValueError('stable_seconds must be finite and nonnegative')
    if query is None:
        query = lambda: query_gpu_state(device=device)

    idle_since = None
    idle_elapsed_seconds = 0.0

    def report(status, snapshot, **extra):
        if on_state is not None:
            on_state(dict(snapshot, status=status, target_mib=target_mib,
                          stable_seconds=stable_seconds,
                          idle_elapsed_seconds=idle_elapsed_seconds, **extra))

    while True:
        try:
            snapshot = _validate_snapshot(query())
        except Exception as error:
            idle_since = None
            idle_elapsed_seconds = 0.0
            report('waiting', {'free_mib': None, 'foreign_pids': []},
                   query_error=f'{type(error).__name__}: {error}')
            sleep(poll_seconds)
            continue
        if snapshot['foreign_pids'] or snapshot['free_mib'] < target_mib:
            idle_since = None
            idle_elapsed_seconds = 0.0
            report('waiting', snapshot)
            sleep(poll_seconds)
            continue
        now = clock()
        if idle_since is None:
            idle_since = now
        idle_elapsed_seconds = max(0.0, now - idle_since)
        if idle_elapsed_seconds < stable_seconds:
            report('waiting', snapshot)
            sleep(poll_seconds)
            continue
        # Import torch only when admission succeeds, never during resource polling.
        if oom_error is None or empty_cache is None:
            import torch
            if oom_error is None:
                oom_error = torch.cuda.OutOfMemoryError
            if empty_cache is None:
                empty_cache = torch.cuda.empty_cache
        report('loading', snapshot)
        try:
            move_to_gpu()
        except oom_error:
            idle_since = None
            idle_elapsed_seconds = 0.0
            rollback_to_cpu()
            empty_cache()
            report('load_oom', snapshot)
            sleep(poll_seconds)
            continue
        # A competing job can start while host weights are being transferred.
        # Recheck admission before doing any model computation. The remaining
        # free memory now includes our allocation, so only foreign PIDs matter.
        try:
            loaded_snapshot = _validate_snapshot(query())
        except Exception as error:
            idle_since = None
            idle_elapsed_seconds = 0.0
            rollback_to_cpu()
            empty_cache()
            report('load_race', snapshot, query_error=f'{type(error).__name__}: {error}')
            sleep(poll_seconds)
            continue
        if loaded_snapshot['foreign_pids']:
            idle_since = None
            idle_elapsed_seconds = 0.0
            rollback_to_cpu()
            empty_cache()
            report('load_race', loaded_snapshot)
            sleep(poll_seconds)
            continue
        report('ready', loaded_snapshot)
        return
