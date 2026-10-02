"""The cross-process GPU gate, against a real Redis (§5.7, D97).

The in-process gate has unit tests with an asyncio heap; those cannot tell this
one anything, because what is under test here is precisely the part that stopped
working when the worker left the API process. §5.7 calls GPU contention "the
single most important resource-contention issue in the design", and M2 measured
what the gate buys (rerank p95 372 → 289 ms for 17% of ingestion throughput,
D79), so the two properties below are the ones those numbers rest on:

  serialisation   two holders never overlap, whatever process they are in;
  ordering        a QUERY waiting behind INGESTION goes first when the slot frees.

Both are asserted against a live Redis rather than a fake, because the mechanism
*is* the Redis semantics — `SET NX PX`, sorted-set ordering and a token-compared
delete. A fake would assert that the test's model of Redis matches the code's
model of Redis, which is not the question.
"""

from __future__ import annotations

import asyncio
import uuid

import pytest
import pytest_asyncio
import redis.asyncio as aioredis

from scholiast.infrastructure.ml.gpu import Priority
from scholiast.infrastructure.ml.redis_gpu import RedisGpuGate

pytestmark = [pytest.mark.integration]

REDIS_URL = "redis://localhost:6379/15"


async def _redis_alive() -> bool:
    try:
        client = aioredis.from_url(REDIS_URL)
        await client.ping()
        await client.aclose()
    except Exception:
        return False
    return True


needs_redis = pytest.mark.skipif(not asyncio.run(_redis_alive()), reason="Redis is not running")


@pytest_asyncio.fixture
async def client():  # type: ignore[no-untyped-def]
    connection = aioredis.from_url(REDIS_URL)
    yield connection
    await connection.aclose()


def _gate(client, **kwargs):  # type: ignore[no-untyped-def]
    # A prefix per test: these run against a shared Redis, and a leftover lock
    # from one test would look like a hung holder in the next.
    return RedisGpuGate(client, prefix=f"test:gpu:{uuid.uuid4().hex[:8]}", **kwargs)


@needs_redis
class TestSerialisation:
    async def test_two_holders_never_overlap(self, client) -> None:  # type: ignore[no-untyped-def]
        """Capacity 1 — the property the whole design rests on."""
        gate = _gate(client)
        overlapping = False
        inside = 0

        async def work() -> None:
            nonlocal overlapping, inside
            async with gate.hold(Priority.INGESTION):
                inside += 1
                if inside > 1:
                    overlapping = True
                await asyncio.sleep(0.02)
                inside -= 1

        await asyncio.gather(*(work() for _ in range(6)))

        assert not overlapping
        assert gate.acquisitions[Priority.INGESTION] == 6

    async def test_separate_gate_objects_still_exclude_each_other(self, client) -> None:  # type: ignore[no-untyped-def]
        # The whole point: two gate INSTANCES, as two processes would have,
        # sharing only Redis. The in-process gate cannot do this at all — and
        # would silently appear to, which is what D97 exists to prevent.
        prefix = f"test:gpu:{uuid.uuid4().hex[:8]}"
        first = RedisGpuGate(client, prefix=prefix)
        second = RedisGpuGate(client, prefix=prefix)
        order: list[str] = []

        async def hold(gate: RedisGpuGate, label: str) -> None:
            async with gate.hold(Priority.INGESTION):
                order.append(f"{label}:in")
                await asyncio.sleep(0.05)
                order.append(f"{label}:out")

        await asyncio.gather(hold(first, "a"), hold(second, "b"))

        # Whoever won, the other one's span must not be nested inside it.
        assert order in (
            ["a:in", "a:out", "b:in", "b:out"],
            ["b:in", "b:out", "a:in", "a:out"],
        ), order

    async def test_the_lock_is_released_when_the_body_raises(self, client) -> None:  # type: ignore[no-untyped-def]
        # The failure that would be worst in production: one raised exception and
        # every later GPU call waits forever with nothing saying why.
        gate = _gate(client)
        with pytest.raises(RuntimeError):
            async with gate.hold(Priority.QUERY):
                raise RuntimeError("boom")

        async with asyncio.timeout(2):
            async with gate.hold(Priority.QUERY):
                pass


@needs_redis
class TestPriority:
    async def test_a_waiting_query_goes_before_waiting_ingestion(self, client) -> None:  # type: ignore[no-untyped-def]
        """§5.7's actual guarantee. Serialisation alone is what CUDA already does."""
        gate = _gate(client)
        served: list[str] = []
        holder_released = asyncio.Event()

        async def holder() -> None:
            async with gate.hold(Priority.INGESTION):
                await holder_released.wait()

        async def waiter(priority: Priority, label: str) -> None:
            async with gate.hold(priority):
                served.append(label)

        first = asyncio.create_task(holder())
        await asyncio.sleep(0.05)  # let the holder take the lock

        # Queue the low-priority waiters FIRST, so arrival order alone would put
        # them ahead. Only priority can reorder them.
        low = [asyncio.create_task(waiter(Priority.INGESTION, f"ingest-{i}")) for i in range(3)]
        await asyncio.sleep(0.05)
        high = asyncio.create_task(waiter(Priority.QUERY, "query"))
        await asyncio.sleep(0.05)

        holder_released.set()
        await asyncio.gather(first, high, *low)

        assert served[0] == "query", served

    async def test_arrival_order_breaks_ties_within_a_priority(self, client) -> None:  # type: ignore[no-untyped-def]
        # FIFO inside a priority, for the reason the in-process gate gives: without
        # it, one paper's batches could interleave with another's in an order
        # nobody chose.
        gate = _gate(client)
        served: list[str] = []
        release = asyncio.Event()

        async def holder() -> None:
            async with gate.hold(Priority.QUERY):
                await release.wait()

        async def waiter(label: str) -> None:
            async with gate.hold(Priority.INGESTION):
                served.append(label)

        first = asyncio.create_task(holder())
        await asyncio.sleep(0.05)

        queued = []
        for index in range(4):
            queued.append(asyncio.create_task(waiter(f"batch-{index}")))
            await asyncio.sleep(0.02)  # distinct arrival times

        release.set()
        await asyncio.gather(first, *queued)

        assert served == ["batch-0", "batch-1", "batch-2", "batch-3"], served


@needs_redis
class TestCrashSafety:
    async def test_the_gate_puts_a_ttl_on_every_lock_it_takes(self, client) -> None:  # type: ignore[no-untyped-def]
        """The TTL has to come from the GATE, which is subtler than it looks.

        The first version of the test below planted a ghost lock with its own
        `px=300` and then watched it expire — which proves Redis expires keys,
        not that this code ever sets an expiry. A mutation that removed the gate's
        TTL entirely passed it. So the expiry is asserted here on a lock the gate
        itself took: without it, a holder killed mid-batch would own the GPU until
        someone cleared Redis by hand.
        """
        prefix = f"test:gpu:{uuid.uuid4().hex[:8]}"
        gate = RedisGpuGate(client, prefix=prefix, ttl_seconds=5)

        async with gate.hold(Priority.QUERY):
            remaining = await client.pttl(f"{prefix}:lock")

        # -1 means "no expiry set", which is the bug; -2 means the key is gone.
        assert 0 < remaining <= 5000, remaining

    async def test_a_holder_that_dies_without_releasing_frees_the_gpu(self, client) -> None:  # type: ignore[no-untyped-def]
        # The crash path end to end, through the gate's own acquire rather than a
        # hand-planted key: a process that is SIGKILLed never runs its release, so
        # only the gate's TTL can give the GPU back.
        prefix = f"test:gpu:{uuid.uuid4().hex[:8]}"
        dead = RedisGpuGate(client, prefix=prefix, ttl_seconds=0.4)
        # The ticket has to be queued before `_wait_turn` can be first in line —
        # it waits for its OWN ticket to reach the front, so an unqueued one spins
        # forever. That is the gate behaving correctly; the first draft of this
        # test skipped the zadd and hung.
        ticket = f"{uuid.uuid4()}"
        await client.zadd(f"{prefix}:waiting", {ticket: 1})
        await dead._wait_turn(ticket, "token-of-a-doomed-process")
        await client.zrem(f"{prefix}:waiting", ticket)
        assert await client.get(f"{prefix}:lock") is not None

        survivor = RedisGpuGate(client, prefix=prefix, ttl_seconds=0.4)
        async with asyncio.timeout(3):
            async with survivor.hold(Priority.QUERY):
                pass

    async def test_a_late_release_cannot_unlock_its_successor(self, client) -> None:  # type: ignore[no-untyped-def]
        # The other half of the TTL bargain. If releasing were an unconditional
        # DELETE, a holder that expired and then finished would unlock the GPU
        # while its successor was still using it — two consumers, silently.
        prefix = f"test:gpu:{uuid.uuid4().hex[:8]}"
        gate = RedisGpuGate(client, prefix=prefix)

        await client.set(f"{prefix}:lock", "successor-token", px=5000)
        await gate._release("expired-holder-token")

        assert await client.get(f"{prefix}:lock") is not None
