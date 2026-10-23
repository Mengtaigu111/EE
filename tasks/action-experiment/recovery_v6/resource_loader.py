"""Bounded same-process GPU reservation before loading a training model.

Reservations contain no compute and never exceed the requested training budget.
They are released into PyTorch's caching allocator immediately before model
transfer, so their memory is reused by model parameters and training activations.
Call after CPU model/dual-adapter construction, before optimizer construction.
No foreign PID gate, stable-idle timer, or monkeypatch of tensor migration is used.
"""
import time
import threading


class MemoryReservation:
    def __init__(self, info, allocate, oom_error, *, target, spare, chunk):
        if target <= 0 or spare < 0 or chunk <= 0:
            raise ValueError('Invalid memory reservation limits')
        self.info, self.allocate, self.oom_error = info, allocate, oom_error
        self.target, self.spare, self.chunk = target, spare, chunk
        self.blocks = []

    def fill(self, *, claim_available=False):
        """Claim currently available budget; never wait or discard prior blocks."""
        while True:
            free, owned = self.info()
            if owned >= self.target or (not claim_available and free + owned - self.spare >= self.target):
                return True
            size = min(free - self.spare, self.target - owned, self.chunk)
            if size <= 0:
                return False
            try:
                block = self.allocate(size)
            except self.oom_error:
                return False
            self.blocks.append(block)

    def release(self):
        # Do not empty_cache: these blocks are the imminent model's budget.
        self.blocks.clear()


class BackgroundReservation:
    """Claim training memory during slow CPU/hash preparation in this process.

    Call ``start()`` before source hashing/model loading, then ``release()``
    immediately before ``wait_for_training_gpu``. Released blocks remain in the
    same CUDA caching allocator; no empty_cache is called. CUDA work is limited
    to bounded allocation. Unexpected worker errors propagate from release().
    """
    def __init__(self, *, target_mib=21000, spare_mib=256, chunk_mib=256,
                 poll_seconds=1.0, memory_info=None, allocate=None,
                 oom_error=None, byte_unit=1024 ** 2, device=0):
        if poll_seconds <= 0 or byte_unit <= 0:
            raise ValueError('poll_seconds and byte_unit must be positive')
        if any(value is None for value in (memory_info, allocate, oom_error)):
            import torch
            if not torch.cuda.is_available():
                raise RuntimeError('CUDA is required for background reservation')
            if memory_info is None:
                memory_info = lambda: (torch.cuda.mem_get_info(device)[0],
                                       torch.cuda.memory_reserved(device))
            if allocate is None:
                allocate = lambda size: torch.empty(size, dtype=torch.uint8,
                                                    device=f'cuda:{device}')
            if oom_error is None:
                oom_error = torch.OutOfMemoryError
        self.reservation = MemoryReservation(memory_info, allocate, oom_error,
            target=int(target_mib * byte_unit), spare=int(spare_mib * byte_unit),
            chunk=int(chunk_mib * byte_unit))
        self.poll_seconds = poll_seconds
        self._stop = threading.Event()
        self._error = None
        self.thread = threading.Thread(target=self._run, daemon=True,
                                       name='training-memory-reservation')

    def _run(self):
        try:
            while not self._stop.is_set():
                # Unlike the transfer gate, CPU preparation needs actual blocks
                # even when the entire desired budget is currently unallocated.
                self.reservation.fill(claim_available=True)
                self._stop.wait(self.poll_seconds)
        except BaseException as error:
            self._error = error

    def start(self):
        self.thread.start()
        return self

    def release(self):
        self._stop.set()
        if self.thread.ident is not None:
            self.thread.join()
        self.reservation.release()
        if self._error is not None:
            raise self._error


def wait_for_training_gpu(move_to_gpu, rollback_to_cpu, *, target_mib=21000,
                          spare_mib=256, chunk_mib=256, poll_seconds=1.0,
                          on_state=None, memory_info=None, allocate=None,
                          oom_error=None, empty_cache=None, sleep=time.sleep,
                          byte_unit=1024 ** 2, device=0):
    """Acquire bounded budget and transfer immediately when it is available.

    ``move_to_gpu`` must finish the complete model transfer; ``rollback_to_cpu``
    must restore every tensor after a partial transfer OOM. Exceptions other
    than the provided OOM type propagate. Neither callback should change dtype
    or replace adapters. Construct the optimizer only after this returns.

    The default CUDA backend has no imports/initialization until called.
    Inject the four backend operations for CPU tests. ``on_state`` receives
    byte counts and status (reserving/loading/load_oom/ready), without a timer.
    """
    if poll_seconds <= 0 or byte_unit <= 0:
        raise ValueError('poll_seconds and byte_unit must be positive')
    if any(x is None for x in (memory_info, allocate, oom_error, empty_cache)):
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA is required for training resource loading')
        if memory_info is None:
            memory_info = lambda: (torch.cuda.mem_get_info(device)[0],
                                   torch.cuda.memory_reserved(device))
        if allocate is None:
            allocate = lambda size: torch.empty(size, dtype=torch.uint8,
                                                device=f'cuda:{device}')
        if oom_error is None:
            oom_error = torch.OutOfMemoryError
        if empty_cache is None:
            empty_cache = torch.cuda.empty_cache
    reservation = MemoryReservation(memory_info, allocate, oom_error,
        target=int(target_mib * byte_unit), spare=int(spare_mib * byte_unit),
        chunk=int(chunk_mib * byte_unit))

    def report(status):
        if on_state is not None:
            free, owned = memory_info()
            on_state(dict(status=status, free_bytes=free, reserved_bytes=owned,
                          target_bytes=reservation.target, spare_bytes=reservation.spare,
                          reservation_blocks=len(reservation.blocks)))

    try:
        while True:
            ready = reservation.fill()
            report('loading' if ready else 'reserving')
            if ready:
                reservation.release()
                try:
                    move_to_gpu()
                except oom_error:
                    rollback_to_cpu()
                    empty_cache()
                    report('load_oom')
                else:
                    report('ready')
                    return
            # Resource polling only: no minimum idle window before transfer.
            sleep(poll_seconds)
    finally:
        reservation.release()
