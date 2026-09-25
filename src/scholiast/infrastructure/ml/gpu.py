"""The GPU priority gate (doc 5 §5.7, D74).

One GPU, two consumers with opposite latency profiles: reranking is a small burst a
user is waiting on, embedding is a sustained background stream. Left unmanaged they
run concurrently and slow each other down — measured at ~3.2x for rerank@50 under
fp16 bulk embedding (D74).

**What this does, and what it cannot do.** Capacity 1, with waiters ordered by
priority and then by arrival. A GPU kernel cannot be interrupted mid-batch, so
"preemption" here means a waiting rerank goes ahead of waiting embedding batches
when the slot frees — it never stops a batch that is already running. A rerank
therefore waits for **at most one embedding batch**, which is why §5.7 pairs this
gate with *bounded* batch sizes. The gate alone is only as good as the batch
duration.

**In-process only.** An `asyncio` primitive coordinates work inside one process.
That holds in M1, where the ingestion worker runs inside the API process (D63), and
stops holding when M3 moves the worker to Celery. That split needs a cross-process
mechanism; this one would silently coordinate nothing.

**Starvation is accepted, not solved.** A continuous stream of high-priority work
would starve low-priority batches. Query traffic is bursty and NFR-P6 allows 90 s
per paper for ingestion, so the trade is deliberate.

Stdlib only. The gate knows nothing about models, tensors or devices — only that
something scarce is being queued for.
"""

from __future__ import annotations

import asyncio
import heapq
import itertools
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from enum import IntEnum


class Priority(IntEnum):
    """Lower value is served first. Two levels because §5.7 names two consumers."""

    QUERY = 0  # reranking: a user is waiting inside NFR-P2's 800 ms budget
    INGESTION = 1  # embedding batches: background, NFR-P6 allows 90 s/paper


class GpuPriorityGate:
    """A capacity-1 lock whose waiters are served by priority, then arrival order.

    FIFO within a priority is deliberate: without the arrival counter as a
    tiebreak, two waiting embedding batches would be ordered arbitrarily by the
    heap, and a paper's batches could interleave with another paper's in an order
    nobody chose.
    """

    def __init__(self) -> None:
        self._busy = False
        # (priority, arrival, future). The arrival counter is unique, so the heap
        # never has to compare futures — which are not orderable.
        self._waiters: list[tuple[int, int, asyncio.Future[None]]] = []
        self._arrivals = itertools.count()
        # Observability for tests and for phase 5's measurement: how often a
        # query actually had to queue, and how many times it jumped ahead.
        self.acquisitions: dict[Priority, int] = dict.fromkeys(Priority, 0)

    @property
    def busy(self) -> bool:
        return self._busy

    @property
    def waiting(self) -> int:
        return sum(1 for _, _, future in self._waiters if not future.done())

    @asynccontextmanager
    async def hold(self, priority: Priority) -> AsyncIterator[None]:
        """Hold the GPU for one unit of work. Released on exit, including on error.

        Release-on-error is the property that matters most: a rerank that raises
        must not leave the gate held, or every later GPU call in the process waits
        forever with nothing in any log to say why.
        """
        await self._acquire(priority)
        try:
            yield
        finally:
            self._release()

    async def _acquire(self, priority: Priority) -> None:
        if not self._busy and not self.waiting:
            self._busy = True
            self.acquisitions[priority] += 1
            return
        loop = asyncio.get_running_loop()
        future: asyncio.Future[None] = loop.create_future()
        heapq.heappush(self._waiters, (int(priority), next(self._arrivals), future))
        try:
            await future
        except asyncio.CancelledError:
            # A cancelled waiter that had ALREADY been handed the slot must pass it
            # on, or the gate is held by a task that no longer exists.
            if future.done() and not future.cancelled():
                self._release()
            raise
        self.acquisitions[priority] += 1

    def _release(self) -> None:
        while self._waiters:
            _, _, future = heapq.heappop(self._waiters)
            if not future.done():
                # Hand the slot straight to the next waiter: `_busy` stays True, so
                # a newcomer arriving in the same tick cannot slip in ahead of a
                # higher-priority task that was already queued.
                future.set_result(None)
                return
        self._busy = False
