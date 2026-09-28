"""Waking a worker for a job that is already durable (§5.4.2, §5.7.4, D95, D96).

Separate from `ports.queue` because the two answer different questions, and
keeping them apart is what makes the durability argument checkable.

  `Queue`           where a job *lives*. Postgres, in the same transaction as
                    the paper and the membership, with a lease and an attempt
                    count that outlive the process holding them.

  `JobDispatcher`   how a worker learns about it *sooner than the next sweep*.

**A dispatch is an optimisation, never a commitment.** By the time it is called
the job row is committed, so the work cannot be lost: an unclaimed job becomes
claimable again when its lease passes, and the periodic sweep finds it. That is
the whole reason Celery can be the transport without giving up D47's
single-transaction guarantee — a broker that cannot join the transaction is not
being asked to. A failed publish costs latency, not work, and the adapter is
therefore allowed to swallow its own errors rather than fail the request that
already succeeded.

**Dispatch happens after the commit, never inside it.** Publishing first would
announce a job that a rollback then erased, and a worker fast enough to claim it
would find nothing — the classic dual-write inversion. The ordering is enforced
where the transaction ends rather than where the job is created, because the use
case does not own the commit.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class JobDispatcher(Protocol):
    """Tell a worker there is something to claim."""

    async def dispatch(self, job_id: str) -> bool:
        """Best-effort wake-up for one committed job.

        Returns whether the broker accepted it, for logging and tests. **Never
        raises for a broker failure**: the caller is past the point where failing
        would help, and the lease already covers the case.
        """
        ...
