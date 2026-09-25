"""The GPU priority gate, tested on the properties §5.7 relies on (D74).

Concurrency code is where a green happy path proves least, so these tests pin the
failure modes that would be silent in production: a query that does NOT jump the
queue, a gate left held after an exception, a cancelled waiter that swallows the
slot. None of those raise. Each shows up only as a rerank that is slow for no
visible reason, or a GPU that is never used again.
"""

from __future__ import annotations

import asyncio

import pytest

from scholiast.infrastructure.ml.gpu import GpuPriorityGate, Priority


async def _hold_until(
    gate: GpuPriorityGate, priority: Priority, release: asyncio.Event, order: list[str], name: str
) -> None:
    async with gate.hold(priority):
        order.append(name)
        await release.wait()


class TestTheGateIsALock:
    async def test_an_uncontended_hold_acquires_and_releases(self) -> None:
        gate = GpuPriorityGate()
        async with gate.hold(Priority.QUERY):
            assert gate.busy
        assert not gate.busy

    async def test_only_one_holder_at_a_time(self) -> None:
        gate = GpuPriorityGate()
        release = asyncio.Event()
        order: list[str] = []
        first = asyncio.create_task(_hold_until(gate, Priority.INGESTION, release, order, "a"))
        await asyncio.sleep(0)
        second = asyncio.create_task(_hold_until(gate, Priority.INGESTION, release, order, "b"))
        await asyncio.sleep(0)
        assert order == ["a"], "a second holder entered while the first still held the gate"
        assert gate.waiting == 1
        release.set()
        await asyncio.gather(first, second)
        assert order == ["a", "b"]


class TestPriorityOrdering:
    async def test_a_query_queued_later_goes_ahead_of_a_waiting_batch(self) -> None:
        # THE property the gate exists for. An ingestion batch is running; another
        # batch queues; THEN a rerank arrives. When the slot frees, the rerank must
        # go next — otherwise a user-facing request waits behind background work
        # and the gate is just a mutex with extra steps.
        gate = GpuPriorityGate()
        order: list[str] = []
        running = asyncio.Event()
        queued = [asyncio.Event() for _ in range(2)]

        async with gate.hold(Priority.INGESTION):
            order.append("batch-1 (running)")
            waiter_batch = asyncio.create_task(
                _hold_until(gate, Priority.INGESTION, queued[0], order, "batch-2")
            )
            await asyncio.sleep(0)
            waiter_query = asyncio.create_task(
                _hold_until(gate, Priority.QUERY, queued[1], order, "rerank")
            )
            await asyncio.sleep(0)
            assert gate.waiting == 2
        running.set()
        await asyncio.sleep(0)
        queued[1].set()
        await asyncio.sleep(0)
        queued[0].set()
        await asyncio.gather(waiter_batch, waiter_query)
        assert order == ["batch-1 (running)", "rerank", "batch-2"]

    async def test_arrival_order_is_kept_within_a_priority(self) -> None:
        # Without the arrival tiebreak the heap orders equal priorities arbitrarily,
        # and one paper's batches could interleave with another's.
        gate = GpuPriorityGate()
        order: list[str] = []
        done = asyncio.Event()
        async with gate.hold(Priority.QUERY):
            tasks = [
                asyncio.create_task(_hold_until(gate, Priority.INGESTION, done, order, f"b{i}"))
                for i in range(5)
            ]
            await asyncio.sleep(0)
        done.set()
        await asyncio.gather(*tasks)
        assert order == ["b0", "b1", "b2", "b3", "b4"]

    async def test_a_handoff_never_lets_a_newcomer_in_alongside_the_waiter(self) -> None:
        # Release hands the slot directly to the next waiter rather than clearing a
        # flag. If it cleared the flag, a task arriving in the same tick would take
        # the fast path and hold the GPU at the SAME TIME as the task it was handed
        # to — two concurrent holders of a capacity-1 resource.
        #
        # Asserted on OCCUPANCY, not order. The first version of this test checked
        # `order == ["query", "newcomer"]` and survived the exact mutation it was
        # written for: asyncio resumes the already-queued task first either way, so
        # the order is identical whether or not the handoff is broken. Only counting
        # how many tasks are inside at once exposes the defect.
        gate = GpuPriorityGate()
        inside = 0
        peak = 0
        release = asyncio.Event()

        async def occupy(priority: Priority) -> None:
            nonlocal inside, peak
            async with gate.hold(priority):
                inside += 1
                peak = max(peak, inside)
                await release.wait()  # stay inside so a broken gate can admit a second
                inside -= 1

        async with gate.hold(Priority.INGESTION):
            queued = asyncio.create_task(occupy(Priority.QUERY))
            await asyncio.sleep(0)
        newcomer = asyncio.create_task(occupy(Priority.QUERY))
        for _ in range(5):
            await asyncio.sleep(0)
        assert peak == 1, f"{peak} tasks held the capacity-1 gate at once"
        release.set()
        await asyncio.gather(queued, newcomer)
        assert peak == 1


class TestTheGateSurvivesFailure:
    async def test_an_exception_inside_the_hold_releases_the_gate(self) -> None:
        # A rerank that raises must not leave the GPU held. Nothing would log it;
        # every later GPU call would simply wait forever.
        gate = GpuPriorityGate()
        with pytest.raises(RuntimeError, match="model exploded"):
            async with gate.hold(Priority.QUERY):
                raise RuntimeError("model exploded")
        assert not gate.busy
        async with asyncio.timeout(1.0):
            async with gate.hold(Priority.INGESTION):
                pass

    async def test_a_waiter_cancelled_before_its_turn_does_not_block_the_next(self) -> None:
        gate = GpuPriorityGate()
        order: list[str] = []
        done = asyncio.Event()
        async with gate.hold(Priority.INGESTION):
            doomed = asyncio.create_task(_hold_until(gate, Priority.QUERY, done, order, "doomed"))
            await asyncio.sleep(0)
            survivor = asyncio.create_task(
                _hold_until(gate, Priority.INGESTION, done, order, "survivor")
            )
            await asyncio.sleep(0)
            doomed.cancel()
            await asyncio.sleep(0)
        done.set()
        async with asyncio.timeout(1.0):
            await survivor
        assert order == ["survivor"]
        assert not gate.busy

    async def test_a_waiter_cancelled_after_being_handed_the_slot_passes_it_on(self) -> None:
        # The subtle case: the slot was handed over, then the task was cancelled
        # before it resumed. If it did not pass the slot on, the gate would be held
        # by a task that no longer exists — permanently.
        gate = GpuPriorityGate()
        order: list[str] = []
        done = asyncio.Event()
        async with gate.hold(Priority.INGESTION):
            handed = asyncio.create_task(_hold_until(gate, Priority.QUERY, done, order, "handed"))
            await asyncio.sleep(0)
            follower = asyncio.create_task(
                _hold_until(gate, Priority.INGESTION, done, order, "follower")
            )
            await asyncio.sleep(0)
        handed.cancel()
        done.set()
        async with asyncio.timeout(1.0):
            await follower
        assert "follower" in order
        assert not gate.busy


class TestObservability:
    async def test_acquisitions_are_counted_per_priority(self) -> None:
        # Phase 5 measures whether the scheme works; it needs to see how often each
        # consumer actually took the GPU.
        gate = GpuPriorityGate()
        async with gate.hold(Priority.QUERY):
            pass
        for _ in range(3):
            async with gate.hold(Priority.INGESTION):
                pass
        assert gate.acquisitions[Priority.QUERY] == 1
        assert gate.acquisitions[Priority.INGESTION] == 3
