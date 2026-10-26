"""Bounded spawn workers with ordered, coordinator-only search commits."""
from collections import Counter, deque
import multiprocessing as mp
import os
import queue
import signal
import time

_DATA = None
_EPOCH = None
_KERNEL = None


class CancelledBatch(Exception):
    """A speculative batch belongs to an abandoned search stage."""


def _initialize(data, epoch, ready, kernel):
    global _DATA, _EPOCH, _KERNEL
    _DATA, _EPOCH, _KERNEL = data, epoch, kernel
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    for value in data.values():
        if hasattr(value, 'setflags'):
            value.setflags(write=False)
    ready.put(os.getpid())


def _kernel(kind, data, payload):
    if kind == 'specular_stage1':
        from .specular import stage1_batch
        return stage1_batch(data, payload)
    if kind == 'specular_stage2':
        from .specular import stage2_batch
        return stage2_batch(data, payload)
    if kind == 'without_specular':
        from .without_specular import seed_batch
        return seed_batch(data, payload)
    raise ValueError(f'Unknown numerical batch: {kind}')


def _execute(task_id, epoch, kind, payload):
    started = time.process_time()
    def check():
        if _EPOCH.value != epoch:
            raise CancelledBatch()
    data = dict(_DATA, check_cancel=check)
    try:
        check()
        result = _KERNEL(kind, data, payload)
        return task_id, result, time.process_time()-started
    except CancelledBatch:
        return task_id, None, time.process_time()-started


class EventCounts(Counter):
    def __init__(self, events):
        super().__init__()
        self.events = events

    def __setitem__(self, key, value):
        delta = value-self[key]
        super().__setitem__(key, value)
        self.events.append((key, delta))


class Events:
    """Record stage-2 outcomes without applying global candidate decisions."""
    def __init__(self):
        self.events = []
        self.counts = EventCounts(self.events)

    def add_pending(self, candidate):
        self.events.append(('candidate', candidate))


class OrderedExecutor:
    def __init__(self, data, workers, kernel=_kernel):
        self.data, self.workers, self.kernel = data, workers, kernel
        self.pool = None
        self.epoch = None
        self.next_id = 0
        self.closed = False
        self.abandoned = False
        self.stale = []
        self.inflight = {}
        self.info = {
            'backend': 'direct' if workers == 1 else 'multiprocessing-spawn',
            'requested_workers': workers, 'started_workers': 0,
            'blas_threads_per_process': 1, 'submitted_tasks': 0,
            'committed_tasks': 0, 'discarded_tasks': 0,
            'max_outstanding_tasks': 0, 'startup_wall_seconds': 0.,
            'coordination_wall_seconds': 0., 'shutdown_wall_seconds': 0.,
            'committed_worker_task_cpu_seconds': 0.,
        }
        self.pids = set()
        self.child_cpu_start = self._child_cpu()

    @staticmethod
    def _child_cpu():
        try:
            import resource
            usage = resource.getrusage(resource.RUSAGE_CHILDREN)
            return usage.ru_utime+usage.ru_stime
        except (ImportError, AttributeError):
            return None

    def start(self, check):
        if self.workers == 1 or self.pool is not None:
            return
        started = time.perf_counter()
        context = mp.get_context('spawn')
        self.epoch = context.Value('q', 0)
        self.ready = context.Queue()
        self.pool = context.Pool(self.workers, initializer=_initialize,
                                 initargs=(self.data, self.epoch, self.ready, self.kernel))
        try:
            while len(self.pids) < self.workers:
                check()
                try:
                    self.pids.add(self.ready.get(timeout=.05))
                    self.info['started_workers'] = len(self.pids)
                except queue.Empty:
                    pass
        finally:
            self.info['startup_wall_seconds'] += time.perf_counter()-started

    def _check_replacement(self):
        # Pool automatically replaces dead workers. Reject this instead of waiting
        # forever for a lost AsyncResult or silently retrying different work.
        try:
            pid = self.ready.get_nowait()
        except queue.Empty:
            return
        raise RuntimeError(f'Numerical worker exited unexpectedly (replacement PID {pid})')

    def ordered(self, kind, tasks, check):
        tasks = iter(tasks)
        if self.workers == 1:
            for payload in tasks:
                check()
                self.info['submitted_tasks'] += 1
                self.next_id += 1
                self.info['max_outstanding_tasks'] = max(self.info['max_outstanding_tasks'], 1)
                try:
                    result = self.kernel(kind, dict(self.data, check_cancel=check), payload)
                    check()
                except BaseException:
                    self.info['discarded_tasks'] += 1
                    raise
                self.info['committed_tasks'] += 1
                yield result
            return
        self.start(check)
        # Drain abandoned stage tasks before submitting another bounded window.
        # Their results are discarded, even if they completed before cancellation.
        for future in self.stale:
            while not future.ready():
                check()
                self._check_replacement()
                future.wait(.05)
        self.stale.clear()
        pending = deque()
        epoch = self.epoch.value
        exhausted = False
        try:
            while pending or not exhausted:
                check()
                started = time.perf_counter()
                try:
                    self._check_replacement()
                    while not exhausted and len(pending) < 2*self.workers:
                        try:
                            payload = next(tasks)
                        except StopIteration:
                            exhausted = True
                            break
                        task_id = self.next_id
                        self.next_id += 1
                        future = self.pool.apply_async(_execute, (task_id, epoch, kind, payload))
                        self.inflight[task_id] = future
                        pending.append((task_id, future))
                        self.info['submitted_tasks'] += 1
                        self.info['max_outstanding_tasks'] = max(self.info['max_outstanding_tasks'], len(pending))
                    if not pending:
                        break
                finally:
                    self.info['coordination_wall_seconds'] += time.perf_counter()-started
                task_id, future = pending[0]
                while not future.ready():
                    check()
                    self._check_replacement()
                    future.wait(.05)
                check()
                started = time.perf_counter()
                try:
                    returned_id, result, cpu = future.get()
                    if returned_id != task_id:
                        raise RuntimeError('Out-of-order numerical batch identifier')
                    pending.popleft()
                    self.inflight.pop(task_id,None)
                    self.info['committed_tasks'] += 1
                    self.info['committed_worker_task_cpu_seconds'] += cpu
                finally:
                    self.info['coordination_wall_seconds'] += time.perf_counter()-started
                yield result
        finally:
            if pending:
                self.abandoned = True
                self.info['discarded_tasks'] += len(pending)
                self.stale.extend(future for _, future in pending)
                # Queued tasks skip execution; running kernels check between
                # candidate refits. The same pool remains available to stage 2.
                with self.epoch.get_lock():
                    self.epoch.value += 1

    def close(self, abort=False):
        if self.closed:
            return
        self.closed = True
        started = time.perf_counter()
        error = None
        if self.pool is not None:
            try:
                # First let cancelled kernels finish and drain their replies.
                # Killing processes while they hold multiprocessing pipe locks
                # risks leaving Pool's own cleanup blocked. No discarded result
                # is committed to the search or its counters.
                drained = True
                if abort or self.abandoned:
                    with self.epoch.get_lock():
                        self.epoch.value += 1
                    end = time.monotonic()+10.
                    for future in self.inflight.values():
                        while not future.ready() and time.monotonic()<end:
                            future.wait(.05)
                        if not future.ready():
                            drained = False
                            break
                self.info['shutdown_mode'] = 'cooperative-drain' if drained else 'terminate-fallback'
                if not drained:
                    self.pool.terminate()
                else:
                    self.pool.close()
                self.pool.join()
            except BaseException as exc:
                # A Ctrl-C arriving during graceful shutdown must also reap workers.
                self.pool.terminate()
                self.pool.join()
                error = exc
            finally:
                self.ready.close()
                self.ready.join_thread()
        self.info['shutdown_wall_seconds'] += time.perf_counter()-started
        end = self._child_cpu()
        self.info['child_process_cpu_seconds'] = (None if end is None or self.child_cpu_start is None
                                                   else max(0., end-self.child_cpu_start))
        self.info['child_cpu_scope'] = ('Reaped child-process CPU, including startup and discarded work; '
                                       'RUSAGE_CHILDREN delta is reliable for an isolated CLI process. '
                                       'Concurrent unrelated child processes in an API caller can contribute.')
        self.info['worker_pids'] = sorted(self.pids)
        if error is not None:
            raise error
