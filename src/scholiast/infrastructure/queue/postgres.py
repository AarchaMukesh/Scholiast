"""Postgres-backed ingestion queue (D47, D48).

`SELECT … FOR UPDATE SKIP LOCKED` over `ingestion_jobs`, which §5.6.3 already
shaped for exactly this: a `visible_at` column and an index on
`(state, visible_at)` labelled "claim query". Doc 7 §7.3 rejected
Postgres-as-queue for production because Celery's redelivery, dead-lettering and
priority semantics would have to be rebuilt by hand — and M3 does swap Celery in.
What it buys in M1 is the property doc 8 asks for and a broker cannot give: the
paper row, the membership row and the job row commit in **one transaction**, so
§5.4.2's "commit the job before enqueueing" has no window left to lose work in.

Four details are deliberate.

**`state` is the pipeline stage, not a generic "in-flight".** The column's CHECK
is already `PaperState`, so a claimed job says *where* a paper is — which is what
`GET /v1/jobs/{id}` and the ops view report (FR-I3, FR-O5).

**The claim orders by priority, then by `visible_at`** (D139). Under D95 the claim
is what decides which job runs, whichever queue the wake-up arrived on, so this
ORDER BY is the whole of "uploads before discovered papers".

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
from collections.abc import Sequence
from datetime import datetime
from typing import Any, cast

from sqlalchemy import ColumnElement, CursorResult, Select, and_, func, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from scholiast.application.ports.queue import (
    DEFAULT_LEASE_SECONDS,
    DEFAULT_MAX_ATTEMPTS,
    DEFAULT_RETRY_BASE_SECONDS,
    MAX_RETRY_AFTER_SECONDS,
    RETRY_JITTER_RATIO,
    JobPriority,
    QueuedJob,
)
from scholiast.application.ports.tenancy import Scope, TenantIsolationViolation
from scholiast.domain.errors import FailureClass
from scholiast.domain.workspace import PaperState, terminal_state_for
from scholiast.infrastructure.db import models, pending
from scholiast.infrastructure.observability.logging import current_correlation_id

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

    async def enqueue(
        self,
        *,
        workspace_id: str,
        paper_id: str,
        priority: JobPriority,
        source_links: Sequence[str] = (),
    ) -> str:
        await self._require_owned_workspace(workspace_id)
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
            # Read from the ambient context rather than taken as a parameter
            # (D107). A correlation id is not a domain concept, so threading it
            # through `AddPaperUseCase` would make the application layer
            # describe logging; None outside a request is the truth.
            correlation_id=current_correlation_id(),
            source_links=list(source_links) or None,
            priority=int(priority),
        )
        self._session.add(row)
        await self._session.flush()
        # Record the wake-up for AFTER this transaction commits (D95). Publishing
        # from here would announce a job a rollback then erased, and a worker
        # quick enough to claim it would find nothing — the dual-write inversion
        # §5.4.2 exists to avoid. `session.info` is the channel because the unit
        # of work belongs to the caller: this adapter knows the job id, and only
        # the composition root knows when the transaction ended.
        key = (
            pending.PENDING_BACKFILL_DISPATCH
            if priority is JobPriority.BACKFILL
            else pending.PENDING_DISPATCH
        )
        pending.record(self._session, key, str(row.id))
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
            # Interactive before backfill (D139), then oldest lease first, so a
            # redelivered job is not starved behind a steady arrival of new ones
            # OF ITS PRIORITY. Strict on purpose: a discovered paper waits while
            # uploads are due, and OLDEST_JOB_AGE{priority} shows how long.
            .order_by(models.IngestionJob.priority.desc(), models.IngestionJob.visible_at)
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
                    # D107. Returned by the claim because that is the only place
                    # the worker learns anything about the job, and the id has to
                    # arrive with the work rather than be fetched afterwards.
                    models.IngestionJob.correlation_id,
                    models.IngestionJob.source_links,
                    models.IngestionJob.priority,
                )
            )
        ).all()
        jobs = [
            QueuedJob(
                id=str(row.id),
                workspace_id=str(row.workspace_id),
                paper_id=str(row.paper_id),
                state=PaperState(row.state),
                attempts=row.attempts,
                correlation_id=row.correlation_id,
                source_links=tuple(row.source_links or ()),
                priority=JobPriority(row.priority),
            )
            for row in rows
        ]
        # `UPDATE ... RETURNING` promises no order, whatever the subquery's ORDER
        # BY said, so a batch would otherwise run a discovered paper before an
        # upload it was claimed with (D139). Within a priority they were all due
        # and claimed together, so their relative order carries nothing.
        return sorted(jobs, key=lambda job: -job.priority)

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
        retry_after: float | None = None,
    ) -> PaperState:
        job = await self._session.scalar(select(models.IngestionJob).where(self._scoped(job_id)))
        if job is None:
            # Nothing to record against. Reporting the terminal state the class
            # implies is still the honest answer for the caller's membership
            # update — the job row's absence does not change what happened.
            return terminal_state_for(failure_class)

        # D142: a wait the failing service stated is honoured, up to an hour.
        # Longer than that and the paper fails now rather than sitting parked.
        too_long = retry_after is not None and retry_after > MAX_RETRY_AFTER_SECONDS
        if too_long:
            detail = (
                f"{detail}; asked to wait {retry_after:.0f} s, longer than the "
                f"{MAX_RETRY_AFTER_SECONDS:.0f} s a retry may wait"
            )
        if failure_class.retryable and job.attempts < max_attempts and not too_long:
            delay = max(retry_delay(job.attempts), retry_after or 0.0)
            await self._apply(
                job_id,
                {
                    "state": PaperState.QUEUED.value,
                    "visible_at": _seconds(delay),
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
            update(models.IngestionJob).where(self._scoped(job_id)).values(**values)
        )
        return bool(cast("CursorResult[Any]", result).rowcount)

    def _scoped(self, job_id: str) -> ColumnElement[bool]:
        """Match one job, and for a tenant scope only one it owns (D92, NFR-T5).

        Closes the gap the M1 log recorded against M3: `heartbeat`, `complete`
        and `fail` matched on the job id alone, so any scope holding an id could
        drive any tenant's job through its state machine. Nothing exposes them
        over HTTP — they are worker-only — but "unreachable today" is not the
        property layer 2 claims, and phase 3 hands them to a Celery fleet.

        A `SystemContext` passes unfiltered: `claim` cannot know whose job it is
        until it has one (`WORKER_CLAIM_REASON`), and the orphaned-job path
        completes a job whose owner could not be resolved at all.
        """
        match = models.IngestionJob.id == uuid.UUID(job_id)
        if self._scope.is_system:
            return match
        return and_(match, models.IngestionJob.workspace_id.in_(self._owned_workspaces()))

    def _owned_workspaces(self) -> Select[tuple[uuid.UUID]]:
        """Workspaces this tenant owns — **including soft-deleted ones (D92)**.

        Deliberately not `ScopedRepository._visible_workspaces()`, and the
        difference is load-bearing rather than stylistic. That one excludes
        workspaces marked DELETING, because §5.6.4 soft-deletes and sweeps
        asynchronously and a deleted workspace must stop serving reads. Here the
        caller is the worker, running under the job owner's `TenantContext`: if a
        user deleted a workspace mid-ingestion, that predicate would stop
        matching, `complete` would change nothing, the lease would expire and the
        job would be re-claimed — a deletion turned into an ingestion loop.

        Scoping by owner keeps a dying workspace's jobs completable, which is
        also right on the merits: the sweeper removes the rows, and a terminal
        state written first is what tells it nothing is still running.
        """
        return select(models.Workspace.id).where(
            models.Workspace.owner_id == uuid.UUID(self._tenant_id())
        )

    async def _require_owned_workspace(self, workspace_id: str) -> None:
        """Refuse to enqueue into a workspace this scope does not own.

        `AddPaperUseCase` has already checked ownership through the workspace
        repository, so a failure here means layer 2 has a hole — which is why it
        raises rather than returning something a caller might branch on. That is
        what `TenantIsolationViolation` is for.

        **This one DOES exclude soft-deleted workspaces, unlike `_scoped`.** The
        asymmetry is the point: finishing work that is already running in a dying
        workspace is correct, and starting new work in one is not. Same table,
        opposite answers, because the two questions differ.
        """
        if self._scope.is_system:
            return
        owned = await self._session.scalar(
            self._owned_workspaces()
            .where(
                models.Workspace.id == uuid.UUID(workspace_id),
                models.Workspace.deleted_at.is_(None),
            )
            .limit(1)
        )
        if owned is None:
            raise TenantIsolationViolation("ingestion_jobs", "enqueue", repr(self._scope))

    def _tenant_id(self) -> str:
        """The tenant's id, read defensively because `Scope` is structural.

        Duplicated from `ScopedRepository` rather than inherited on purpose:
        inheriting would bring `_visible_workspaces()` into scope here, and the
        docstring above explains why using it would be a bug.
        """
        user_id = getattr(self._scope, "user_id", None)
        if not isinstance(user_id, str) or not user_id:
            raise TenantIsolationViolation("Scope", repr(self._scope), "missing user_id")
        return user_id
