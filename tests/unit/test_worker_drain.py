"""One wake-up drains what is due, inside a budget (D141).

Without the drain a wake-up claims one job, so a backlog of retries in the Celery
topology clears at one per sweep. Without the budget a long drain would outlive
its own message's visibility timeout and Redis would hand the wake-up to another
worker.
"""

from __future__ import annotations

from scholiast.workers.ingestion import IngestionWorker, WorkerConfig


class Clock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


def worker(
    script: list[int], *, lease: float = 10.0, clock: Clock | None = None, step: float = 0.0
) -> tuple[IngestionWorker, list[int]]:
    timer = clock or Clock()
    built = IngestionWorker(
        sessions=None,  # type: ignore[arg-type]
        store=None,  # type: ignore[arg-type]
        parser=None,  # type: ignore[arg-type]
        embedder=None,  # type: ignore[arg-type]
        index=None,  # type: ignore[arg-type]
        config=WorkerConfig(lease_seconds=lease),
        timer=timer,
    )
    calls: list[int] = []

    async def run_once() -> int:
        calls.append(1)
        timer.now += step
        return script.pop(0) if script else 0

    built.run_once = run_once  # type: ignore[method-assign]
    return built, calls


async def test_a_wake_up_keeps_claiming_until_nothing_is_due() -> None:
    drainer, calls = worker([1, 1, 1, 0])

    assert await drainer.drain() == 3
    assert len(calls) == 4


async def test_an_empty_queue_costs_one_claim() -> None:
    drainer, calls = worker([0])

    assert await drainer.drain() == 0
    assert len(calls) == 1


async def test_no_job_starts_once_half_the_lease_has_passed() -> None:
    # Each job takes 3 s against a 10 s lease: jobs start at 0 s and 3 s, and
    # at 6 s the 5 s budget is spent, so the rest waits for the next wake-up.
    clock = Clock()
    drainer, calls = worker([1] * 10, lease=10.0, clock=clock, step=3.0)

    assert await drainer.drain() == 2
    assert len(calls) == 2
