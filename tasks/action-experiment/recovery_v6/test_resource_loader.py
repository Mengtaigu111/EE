"""CPU resource-policy tests; importing this suite never initializes CUDA."""
import unittest
import threading

from resource_loader import BackgroundReservation, MemoryReservation, wait_for_training_gpu


class OOM(RuntimeError):
    pass


class Backend:
    def __init__(self, free):
        self.free, self.owned = free, 0
        self.fail = False
        self.allocations = []

    def info(self):
        return self.free, self.owned

    def allocate(self, size):
        if self.fail:
            raise OOM('race')
        self.free -= size
        self.owned += size
        self.allocations.append(size)
        return object()


class ResourceTests(unittest.TestCase):
    def reservation(self, b):
        return MemoryReservation(b.info, b.allocate, OOM, target=100, spare=5, chunk=20)

    def test_claims_new_capacity_without_stability_window(self):
        b = Backend(45)
        r = self.reservation(b)
        self.assertFalse(r.fill())
        self.assertEqual(b.owned, 40)
        b.free += 60
        self.assertTrue(r.fill())

    def test_immediate_transfer_when_budget_already_available(self):
        b = Backend(105)
        events = []
        result = wait_for_training_gpu(lambda: events.append('transfer'), lambda: events.append('rollback'),
            memory_info=b.info, allocate=b.allocate, oom_error=OOM,
            empty_cache=lambda: None, sleep=lambda s: self.fail('unnecessary wait'),
            target_mib=100, spare_mib=5, chunk_mib=20, byte_unit=1)
        self.assertEqual(events, ['transfer'])
        self.assertIsNone(result)
        self.assertEqual(b.allocations, [])

    def test_transient_allocation_oom_keeps_prior_blocks(self):
        b = Backend(45)
        r = self.reservation(b)
        self.assertFalse(r.fill())
        before = list(r.blocks)
        b.free += 20
        b.fail = True
        self.assertFalse(r.fill())
        self.assertEqual(r.blocks, before)

    def test_insufficient_memory_only_polls_then_transfers(self):
        b = Backend(45)
        sleeps, events = [], []
        def sleep(seconds):
            sleeps.append(seconds)
            b.free += 60
        wait_for_training_gpu(lambda: events.append('transfer'), lambda: None,
            memory_info=b.info, allocate=b.allocate, oom_error=OOM,
            empty_cache=lambda: None, sleep=sleep,
            target_mib=100, spare_mib=5, chunk_mib=20, byte_unit=1)
        self.assertEqual(sleeps, [1.0])
        self.assertEqual(events, ['transfer'])

    def test_failed_transfer_rolls_back_before_retry(self):
        b = Backend(105)
        events = []
        def transfer():
            events.append('transfer')
            if events.count('transfer') == 1:
                raise OOM('partial move')
        wait_for_training_gpu(transfer, lambda: events.append('rollback'),
            memory_info=b.info, allocate=b.allocate, oom_error=OOM,
            empty_cache=lambda: events.append('empty'), sleep=lambda s: events.append('poll'),
            target_mib=100, spare_mib=5, chunk_mib=20, byte_unit=1)
        self.assertEqual(events, ['transfer', 'rollback', 'empty', 'poll', 'transfer'])

    def test_non_oom_is_not_silently_retried(self):
        b = Backend(105)
        def fail():
            raise ValueError('bad model')
        with self.assertRaisesRegex(ValueError, 'bad model'):
            wait_for_training_gpu(fail, lambda: None, memory_info=b.info,
                allocate=b.allocate, oom_error=OOM, empty_cache=lambda: None,
                sleep=lambda s: self.fail('should propagate'),
                target_mib=100, spare_mib=5, chunk_mib=20, byte_unit=1)

    def test_reservations_are_bounded(self):
        b = Backend(1000)
        r = self.reservation(b)
        self.assertTrue(r.fill())
        self.assertEqual(b.owned, 0)
        for bad in (0, -1):
            with self.assertRaises(ValueError):
                MemoryReservation(b.info, b.allocate, OOM, target=bad, spare=5, chunk=20)

    def test_background_reserves_immediately_and_release_keeps_allocator_cache(self):
        b = Backend(105)
        claimed = threading.Event()
        def allocate(size):
            block = b.allocate(size)
            if b.owned == 100:
                claimed.set()
            return block
        reservation = BackgroundReservation(target_mib=100, spare_mib=5, chunk_mib=20,
            byte_unit=1, memory_info=b.info, allocate=allocate, oom_error=OOM).start()
        try:
            self.assertTrue(claimed.wait(2), 'should allocate before any polling wait')
        finally:
            reservation.release()
        self.assertEqual(b.owned, 100)
        self.assertEqual(reservation.reservation.blocks, [])
        self.assertFalse(reservation.thread.is_alive())
        self.assertTrue(reservation.thread.daemon)

    def test_background_exception_propagates_on_release(self):
        called = threading.Event()
        def broken_info():
            called.set()
            raise ValueError('broken memory query')
        reservation = BackgroundReservation(memory_info=broken_info,
            allocate=lambda size: object(), oom_error=OOM).start()
        self.assertTrue(called.wait(2))
        with self.assertRaisesRegex(ValueError, 'broken memory query'):
            reservation.release()
        self.assertFalse(reservation.thread.is_alive())


if __name__ == '__main__':
    unittest.main()
