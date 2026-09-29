"""Work left on the session for the composition root to dispatch after commit.

**The channel, not the payload.** An adapter knows *that* something was
committed and needs a wake-up; only the caller that owns the unit of work knows
*when* the transaction ended. `session.info` carries the fact across that gap
without either side importing the other (D95, D99).

Publishing from inside the transaction would announce work a rollback then
erased, and a worker quick enough to act on it would find nothing — the
dual-write inversion §5.4.2 exists to avoid. Everything here is therefore
drained *after* the commit, and a drain that fails costs latency rather than
work: an undispatched job is found by its lease, an undispatched deletion by
`list_deleting` on the next reconciliation pass.

The keys are module constants rather than literals at both ends because a typo
in one of them means "nothing was ever dispatched", with no error anywhere.
"""

from __future__ import annotations

from sqlalchemy.ext.asyncio import AsyncSession

#: Ingestion job ids left by `PostgresQueue.enqueue` (D95).
PENDING_DISPATCH = "scholiast.pending_job_dispatch"

#: Workspace ids left by `SqlWorkspaceRepository.soft_delete` (D99). A separate
#: key rather than a tagged entry in the one above: the two are dispatched to
#: different tasks on different queues, and a single list of bare strings would
#: let a workspace id be published as an ingestion job with nothing to catch it.
PENDING_DELETION_DISPATCH = "scholiast.pending_deletion_dispatch"


def record(session: AsyncSession, key: str, value: str) -> None:
    """Note one id for dispatch once this session's transaction commits."""
    pending: list[str] = session.info.setdefault(key, [])
    pending.append(value)


def drain(session: AsyncSession, key: str) -> list[str]:
    """Take everything recorded under `key`, leaving nothing behind.

    Popped rather than read so a second drain of the same session cannot
    re-dispatch what the first one already published.
    """
    drained: list[str] = session.info.pop(key, [])
    return drained
