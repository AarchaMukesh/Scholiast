"""The cross-process GPU gate (§5.7, D97).

`GpuPriorityGate` says it of itself: an asyncio primitive "holds in M1, where the
ingestion worker runs inside the API process (D63), and stops holding when M3
moves the worker to Celery. That split needs a cross-process mechanism; this one
would silently coordinate nothing." This is that mechanism, and it keeps the same
`hold(priority)` contract so the call sites do not learn which topology they are
in — the embedder and the reranker are handed a gate, not a deployment.

**Why Redis and not a file lock or a CUDA primitive.** Redis is already the
broker, so this adds no dependency and fails in the same direction as the queue:
if Redis is unreachable there is no Celery worker to contend with, and the API
degrades to the uncontended case it was measured in. A file lock would be
confined to one host, which is the wrong boundary for a design that scales
workers horizontally (§5.7's growth path), and CUDA's own serialisation orders
kernels without any notion of which caller a user is waiting on.

**Two structures, because one cannot express both facts.**

  `<prefix>:lock`     a `SET NX PX` mutex holding a token unique to its owner.
                      Capacity 1 — the GPU.

  `<prefix>:waiting`  a sorted set of tickets scored `priority * 1e12 + arrival`,
                      so QUERY sorts ahead of INGESTION and arrival breaks ties
                      inside a priority, exactly as the in-process heap does.

A waiter may take the lock only when it is the lowest-scored ticket. Without that
check the mutex would be a free-for-all and the *priority* half of §5.7 — the
half that matters, since capacity-1 serialisation is what CUDA gives anyway —
would be decoration.

**The TTL is a crash guard, not a timeout.** A holder that dies takes its lock
with it unless the key expires; the TTL is therefore comfortably longer than the
longest legitimate hold (one embedding batch, ~0.5 s at batch 32; one rerank@50,
~160 ms measured). A holder still alive past the TTL is a bug this cannot fix —
it releases by token comparison, so it will refuse to delete a lock that has been
handed to someone else rather than silently unlocking a live holder.

**Polling is accepted, and bounded by what it costs.** Waiters spin at
`poll_seconds` rather than blocking, because a blocking `BLPOP` cannot be ordered
by priority — the releaser would have to decide the next owner, and a
higher-priority waiter arriving one millisecond later would sit behind a lower
one. At a 5 ms poll against holds of 160-500 ms, the ordering is worth more than
the jitter, and the wait is idle rather than busy. D97 requires the contention
numbers to be re-measured across processes rather than carried over, so what this
costs is reported rather than argued.
"""

from __future__ import annotations

import asyncio
import time
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

from scholiast.infrastructure.ml.gpu import Priority

#: Longer than any legitimate hold, short enough that a crashed holder does not
#: stall ingestion for a noticeable time. See the module docstring.
DEFAULT_TTL_SECONDS = 30.0

#: 5 ms against holds of 160-500 ms: under 3% of the shortest hold.
DEFAULT_POLL_SECONDS = 0.005

#: Scores are `priority * PRIORITY_BAND + arrival`. The band is wide enough that
#: no arrival counter can ever climb into the next priority's range.
PRIORITY_BAND = 1_000_000_000_000

#: Releasing compares the token before deleting, so a holder whose TTL expired
#: cannot delete the lock its successor now holds. Lua because that comparison
#: and the delete have to be one atomic step.
_RELEASE = """
if redis.call('get', KEYS[1]) == ARGV[1] then
    return redis.call('del', KEYS[1])
end
return 0
"""


def build_redis_client(url: str) -> Any:
    """The raw client the gate needs, constructed where the vendor is known.

    Celery keeps its own connection pool for the broker; this is a second, small
    one for the gate. They are deliberately not shared: the broker's pool is
    sized and configured for task traffic, and a gate that spins at 5 ms would
    contend with it for connections at exactly the moment the GPU is busy.
    """
    import redis.asyncio as aioredis

    return aioredis.from_url(url)


class RedisGpuGate:
    """A capacity-1, priority-ordered gate shared by every process on the GPU."""

    def __init__(
        self,
        client: Any,
        *,
        prefix: str = "scholiast:gpu",
        ttl_seconds: float = DEFAULT_TTL_SECONDS,
        poll_seconds: float = DEFAULT_POLL_SECONDS,
    ) -> None:
        self._redis = client
        self._lock_key = f"{prefix}:lock"
        self._waiting_key = f"{prefix}:waiting"
        self._ttl_ms = int(ttl_seconds * 1000)
        self._poll = poll_seconds
        #: Same counters the in-process gate exposes, so the contention harness
        #: reads one interface whichever gate it was handed.
        self.acquisitions: dict[Priority, int] = dict.fromkeys(Priority, 0)

    @asynccontextmanager
    async def hold(self, priority: Priority) -> AsyncIterator[None]:
        """Hold the GPU for one unit of work. Released on exit, including on error.

        The ticket is removed in the same `finally` as the lock: a waiter that is
        cancelled — a dropped HTTP request, a worker shutting down — must not
        leave its ticket in the sorted set, or every lower-priority waiter would
        queue behind a ticket whose owner is gone until the ticket ages out.
        """
        ticket = f"{uuid.uuid4()}"
        token = f"{uuid.uuid4()}"
        score = priority.value * PRIORITY_BAND + time.monotonic_ns()
        await self._redis.zadd(self._waiting_key, {ticket: score})
        try:
            await self._wait_turn(ticket, token)
            self.acquisitions[priority] += 1
            try:
                yield
            finally:
                await self._release(token)
        finally:
            await self._redis.zrem(self._waiting_key, ticket)

    async def _wait_turn(self, ticket: str, token: str) -> None:
        """Spin until this ticket is first in line AND the lock is free."""
        while True:
            first = await self._redis.zrange(self._waiting_key, 0, 0)
            if first and _as_text(first[0]) == ticket:
                acquired = await self._redis.set(self._lock_key, token, nx=True, px=self._ttl_ms)
                if acquired:
                    return
            await asyncio.sleep(self._poll)

    async def _release(self, token: str) -> None:
        await self._redis.eval(_RELEASE, 1, self._lock_key, token)


def _as_text(value: object) -> str:
    """Redis clients return bytes or str depending on `decode_responses`."""
    return value.decode() if isinstance(value, bytes | bytearray) else str(value)
