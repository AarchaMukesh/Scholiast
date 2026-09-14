"""Transaction boundary port.

§5.4.2 says of the ingestion worker: "Each stage: persist state transition, emit
metric, propagate correlation ID." *Persist* is the load-bearing word. A pipeline
that writes every stage transition inside one long transaction makes them all
visible at the same instant — the end — so a paper reads as QUEUED for ninety
seconds and then INDEXED, and the lease extensions that are supposed to keep the
job claimed are equally invisible until the work they were protecting is over.

So the use case has to be able to say "this much is durable now" without knowing
what a session is. One method, deliberately: this is not a repository registry or
a nested-transaction abstraction, it is a named commit point. Rollback stays with
the adapter, which owns the session and is the only thing that can honour it.

The stage boundaries are also where the retry story lives. Committing per stage
means a crash leaves a *prefix* of the work durable and the job claimable again;
because section replacement is idempotent (D52) and chunk and point ids are
deterministic (D46), re-running that prefix converges rather than duplicating.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class UnitOfWork(Protocol):
    """A named durability boundary inside a multi-stage use case."""

    async def commit(self) -> None:
        """Make everything written so far durable and visible to other readers."""
        ...
