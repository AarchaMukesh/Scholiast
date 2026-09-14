"""Postgres-backed ingestion queue (D47, D48).

`SELECT … FOR UPDATE SKIP LOCKED` over `ingestion_jobs`, which §5.6.3 already
shaped for exactly this: a `visible_at` column and an index on
`(state, visible_at)` labelled "claim query". Doc 7 §7.3 rejected
Postgres-as-queue for production because Celery's redelivery, dead-lettering and
priority semantics would have to be rebuilt by hand — and M3 does swap Celery in.
What it buys in M1 is the property doc 8 asks for and a broker cannot give: the
paper row, the membership row and the job row commit in **one transaction**, so
§5.4.2's "commit the job before enqueueing" has no window left to lose work in.

Three details are deliberate.

**`state` is the pipeline stage, not a generic "in-flight".** The column's CHECK
is already `PaperState`, so a claimed job says *where* a paper is — which is what
`GET /v1/jobs/{id}` and the ops view report (FR-I3, FR-O5).

**`attempts` counts deliveries, incremented on claim rather than on failure.**
A worker that is OOM-killed mid-job never reports anything, so a counter that
only advances on a reported failure would redeliver a poison job forever. Counting
deliveries bounds that case with the same number.

**All time arithmetic is the database's.** The claim predicate and the lease are
evaluated by one clock; computing `visible_at` in Python would make redelivery
depend on the app and database clocks agreeing, which is a bug that appears only
under skew and looks like a queue that drops or duplicates jobs.
"""

from __future__ import annotations

import random
import uuid
from datetime import datetime
from typing import Any, cast

from sqlalchemy import ColumnElement, CursorResult, func, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from scholiast.application.ports.queue import (
    DEFAULT_LEASE_SECONDS,
    DEFAULT_MAX_ATTEMPTS,
    DEFAULT_RETRY_BASE_SECONDS,
    RETRY_JITTER_RATIO,
    QueuedJob,
)
from scholiast.application.ports.tenancy import Scope, TenantIsolationViolation
from scholiast.domain.errors import FailureClass
from scholiast.domain.workspace import PaperState, terminal_state_for
from scholiast.infrastructure.db import models

#: Claimable states: queued, plus every in-flight stage. Including the in-flight
#: stages is what makes the lease a redelivery mechanism — a worker that died
#: while PARSING left the row saying PARSING, and that row must come back.
#: Terminal states are absent, so a dead-lettered job is never re-claimed.
CLAIMABLE: tuple[PaperState, ...] = (
    PaperState.QUEUED,
    PaperState.FETCHING,
    PaperState.PARSING,
    PaperState.CHUNKING,
    PaperState.EMBEDDING,
)

#: `last_error` is a diagnostic, not a log. Truncating keeps one pathological
#: stack trace from making the jobs table expensive to scan.
MAX_ERROR_CHARS = 2000


def _seconds(value: float) -> ColumnElement[datetime]:
    """`now() + value seconds`, computed by Postgres."""
    return func.now() + func.make_interval(0, 0, 0, 0, 0, 0, value)


def retry_delay(attempts: int, *, base: float = DEFAULT_RETRY_BASE_SECONDS) -> float:
    """Exponential backoff with jitter (D48, NFR-R3).

    `attempts` is the delivery count, so the first retry waits `base`. Jitter is
    proportional rather than absolute: it has to stay meaningful as the delay
    grows, and ±20% of 5 s and of 40 s are both about a fifth of the wait.
    """
    # `2 ** n` types as Any for a non-literal exponent (it could be negative
    # and yield a float), so the annotation is what keeps this function honest
    # about returning a number rather than whatever that expression produced.
    span: float = base * float(2 ** max(attempts - 1, 0))
    # The S311 suppression below is deliberate: this is scheduling jitter, not
    # a security decision, and a cryptographic source would cost entropy for a
    # property nothing here needs.
    jitter: float = random.uniform(-RETRY_JITTER_RATIO, RETRY_JITTER_RATIO)  # noqa: S311
    return span * (1.0 + jitter)


class PostgresQueue:
    """The `Queue` adapter.

    Constructed with a session, exactly as the repositories are: that is what
    lets `enqueue` land in the same transaction as the rows the job refers to.
    """

    def __init__(self, session: AsyncSession, scope: Scope) -> None:
        self._session = session
        self._scope = scope

    async def enqueue(self, *, workspace_id: str, paper_id: str) -> str:
        row = models.IngestionJob(
            id=uuid.uuid4(),
            # Malformed ids here are a programming error, not user input: these
            # come from rows this transaction just created. Letting `UUID()`
            # raise is right, unlike on the read paths where an id arrives from
            # a URL and must degrade to "not found".
            workspace_id=uuid.UUID(workspace_id),
            paper_id=uuid.UUID(paper_id),
            state=PaperState.QUEUED.value,
            attempts=0,
        )
        self._session.add(row)
        await self._session.flush()
        return str(row.id)

    async def claim(
        self, *, limit: int = 1, lease_seconds: float = DEFAULT_LEASE_SECONDS
    ) -> list[QueuedJob]:
        if not self._scope.is_system:
            # The claim is the one genuinely cross-tenant step (D50) — a worker
            # cannot know whose job it is until it has one. Refusing a tenant
            # scope here keeps that from becoming a convenient way to read the
            # whole queue: everything *after* the claim runs under the owner's
            # context, and this is the line between the two.
            raise TenantIsolationViolation("ingestion_jobs", "claim", repr(self._scope))
        due = (
            select(models.IngestionJob.id)
            .where(
                models.IngestionJob.state.in_([s.value for s in CLAIMABLE]),
                models.IngestionJob.visible_at <= func.now(),
            )
            # Oldest lease first, so a redelivered job is not starved behind a
            # steady arrival of new ones.
            .order_by(models.IngestionJob.visible_at)
            .limit(limit)
            # The whole mechanism: a row another worker has claimed in a
            # concurrent transaction is skipped rather than waited on, so two
            # workers are a throughput change and never a correctness one.
            .with_for_update(skip_locked=True)
        )
        rows = (
            await self._session.execute(
                update(models.IngestionJob)
                .where(models.IngestionJob.id.in_(due.scalar_subquery()))
                .values(
                    visible_at=_seconds(lease_seconds),
                    attempts=models.IngestionJob.attempts + 1,
                )
                .returning(
                    models.IngestionJob.id,
                    models.IngestionJob.workspace_id,
                    models.IngestionJob.paper_id,
                    models.IngestionJob.state,
                    models.IngestionJob.attempts,
                )
            )
        ).all()
        return [
            QueuedJob(
                id=str(row.id),
                workspace_id=str(row.workspace_id),
                paper_id=str(row.paper_id),
                state=PaperState(row.state),
                attempts=row.attempts,
            )
            for row in rows
        ]

    async def heartbeat(
        self, job_id: str, *, state: PaperState, lease_seconds: float = DEFAULT_LEASE_SECONDS
    ) -> bool:
        # `last_error` is left alone: it records the last thing that went wrong,
        # and a job making progress after a retry has not unmade that.
        return await self._apply(
            job_id, {"state": state.value, "visible_at": _seconds(lease_seconds)}
        )

    async def complete(self, job_id: str) -> bool:
        return await self._apply(job_id, {"state": PaperState.INDEXED.value, "last_error": None})

    async def fail(
        self,
        job_id: str,
        *,
        failure_class: FailureClass,
        detail: str,
        max_attempts: int = DEFAULT_MAX_ATTEMPTS,
    ) -> PaperState:
        job = await self._session.get(models.IngestionJob, uuid.UUID(job_id))
        if job is None:
            # Nothing to record against. Reporting the terminal state the class
            # implies is still the honest answer for the caller's membership
            # update — the job row's absence does not change what happened.
            return terminal_state_for(failure_class)

        if failure_class.retryable and job.attempts < max_attempts:
            await self._apply(
                job_id,
                {
                    "state": PaperState.QUEUED.value,
                    "visible_at": _seconds(retry_delay(job.attempts)),
                    "last_error": detail[:MAX_ERROR_CHARS],
                },
            )
            return PaperState.QUEUED

        # Exhausted retries land on FAILED — §4.4.1 offers a manual retry there.
        # A class retrying cannot help lands wherever the taxonomy says, which is
        # UNSUPPORTED for a file that is what it is and FAILED where a user
        # action (uploading the PDF) is the remedy.
        landed = PaperState.FAILED if failure_class.retryable else terminal_state_for(failure_class)
        await self._apply(job_id, {"state": landed.value, "last_error": detail[:MAX_ERROR_CHARS]})
        return landed

    async def _apply(self, job_id: str, values: dict[str, object]) -> bool:
        result = await self._session.execute(
            update(models.IngestionJob)
            .where(models.IngestionJob.id == uuid.UUID(job_id))
            .values(**values)
        )
        return bool(cast("CursorResult[Any]", result).rowcount)
