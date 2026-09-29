"""The in-process ingestion worker (doc 8 M1, §5.7.3, D47, D50).

Claim a job, run the pipeline, record where it landed. Doc 8 asks for an
in-process queue in M1 and Celery in M3; what makes that a *transport* swap
rather than a rewrite is that everything interesting lives either in the use
case (application) or in the job table (§5.6.3), and this file only carries the
loop.

**Two scopes, and the boundary between them is the point** (D50). The claim is
genuinely cross-tenant — a worker cannot know whose job it is until it has one —
so it runs under `SystemContext` with a named reason, keeping the full set of
cross-tenant operations greppable. Everything after that runs under the owning
user's `TenantContext`, so every write the pipeline makes goes through layer 2's
predicate. Running the whole worker as system would be one query cheaper and
would take the primary isolation layer offline across the largest write path in
the system.

**Each job gets its own session.** The claim commits immediately so the lease is
durable and the row lock is released — otherwise a job would be protected by an
open transaction rather than by the visibility timeout, and a crashed worker's
row would unlock instantly instead of being redelivered on the schedule the
design specifies.
"""

from __future__ import annotations

import asyncio
import contextlib
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime

import structlog
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from scholiast.application.ports.ml import Embedder
from scholiast.application.ports.parsing import DocumentParser
from scholiast.application.ports.queue import DEFAULT_LEASE_SECONDS, QueuedJob
from scholiast.application.ports.storage import ObjectStore
from scholiast.application.ports.tenancy import WORKER_CLAIM_REASON, SystemContext, TenantContext
from scholiast.application.ports.vector import VectorIndex
from scholiast.application.use_cases.ingest_paper import IngestionOutcome, IngestPaperUseCase
from scholiast.domain.chunking import ChunkingConfig
from scholiast.domain.errors import FailureClass, IngestionFailure
from scholiast.domain.workspace import PaperMembership, PaperState
from scholiast.infrastructure.db import models
from scholiast.infrastructure.db.engine import SqlUnitOfWork
from scholiast.infrastructure.db.repositories import (
    SqlChunkRepository,
    SqlPaperRepository,
    SqlSectionRepository,
)
from scholiast.infrastructure.queue.postgres import PostgresQueue

log = structlog.get_logger(__name__)


@dataclass(frozen=True, slots=True)
class WorkerConfig:
    """How hard the loop tries and how long it waits.

    `poll_seconds` is the cost of not having a broker: with the queue in
    Postgres there is nothing to push, so the loop asks. One second is
    imperceptible against NFR-P6's 90 s ingestion budget and cheap against an
    indexed query that usually returns nothing.
    """

    poll_seconds: float = 1.0
    lease_seconds: float = DEFAULT_LEASE_SECONDS
    batch: int = 1
    embed_batch_size: int | None = None
    chunking: ChunkingConfig | None = None


class _FailureRecordingFailed(Exception):
    """A classified failure that could not be written to its own session.

    Carries the outcome so the crash handler can record the ORIGINAL class and
    detail on a fresh session. Private because it never escapes this module: it
    exists to move one value across a `raise`, not to describe a domain event.
    """

    def __init__(self, outcome: IngestionOutcome, cause: Exception) -> None:
        super().__init__(str(cause))
        self.outcome = outcome


class IngestionWorker:
    """Drains the ingestion queue until stopped."""

    def __init__(
        self,
        *,
        sessions: async_sessionmaker[AsyncSession],
        store: ObjectStore,
        parser: DocumentParser,
        embedder: Embedder,
        index: VectorIndex,
        config: WorkerConfig | None = None,
        clock: Callable[[], datetime] = lambda: datetime.now(UTC),
    ) -> None:
        self._sessions = sessions
        self._store = store
        self._parser = parser
        self._embedder = embedder
        self._index = index
        self._config = config or WorkerConfig()
        self._clock = clock
        self._stopping = asyncio.Event()

    # -- loop --------------------------------------------------------------

    async def run_forever(self) -> None:
        """Poll until `stop()` is called. Started from the API's lifespan."""
        while not self._stopping.is_set():
            processed = await self.run_once()
            if processed:
                continue
            # Nothing due. Wait, but wake immediately on shutdown rather than
            # making the process hang for a poll interval on exit.
            with contextlib.suppress(TimeoutError):
                await asyncio.wait_for(self._stopping.wait(), self._config.poll_seconds)

    def stop(self) -> None:
        self._stopping.set()

    async def run_once(self) -> int:
        """Claim and process one batch. Returns how many jobs ran.

        Exposed rather than private because it is what tests drive: a loop that
        can only be started and stopped is a loop whose behaviour has to be
        observed through timing, which is how flaky tests are written.
        """
        jobs = await self._claim()
        for job in jobs:
            await self._process(job)
        return len(jobs)

    # -- one job -----------------------------------------------------------

    async def _claim(self) -> list[QueuedJob]:
        """Take due work. Cross-tenant by necessity, and only this step is.

        Committed immediately: the lease has to be durable and the row lock has
        to be released, or the job would be held by an open transaction for its
        entire run and a crash would make it claimable instantly rather than
        after the visibility timeout.
        """
        system = SystemContext(reason=WORKER_CLAIM_REASON)
        async with self._sessions() as session:
            jobs = await PostgresQueue(session, system).claim(
                limit=self._config.batch, lease_seconds=self._config.lease_seconds
            )
            await session.commit()
            return jobs

    async def _process(self, job: QueuedJob) -> IngestionOutcome | None:
        owner_id = await self._owner_of(job.workspace_id)
        if owner_id is None:
            # The workspace was deleted under the job. Nothing to ingest into
            # and nobody to tell, so the job is retired rather than retried.
            async with self._sessions() as session:
                await PostgresQueue(session, SystemContext(reason=WORKER_CLAIM_REASON)).complete(
                    job.id
                )
                await session.commit()
            log.warning("ingestion.orphaned_job", job_id=job.id, workspace_id=job.workspace_id)
            return None

        # From here on, layer 2 is back in force: every repository below is
        # built with the owning user's scope (D50).
        scope = TenantContext(user_id=owner_id)
        try:
            return await self._run(job, scope)
        except Exception as exc:
            # A worker that dies on one job stops ingesting for everybody, so
            # nothing escapes this loop. The stack is logged rather than
            # swallowed, and the job goes through the ordinary retry path: a
            # genuine bug then dead-letters after the bounded attempts instead
            # of being redelivered forever (D48).
            log.exception("ingestion.crashed", job_id=job.id, paper_id=job.paper_id)
            await self._abandon(job, scope, exc)
            return None

    async def _run(self, job: QueuedJob, scope: TenantContext) -> IngestionOutcome:
        async with self._sessions() as session:
            queue = PostgresQueue(session, scope)
            papers = SqlPaperRepository(session, scope)
            membership = PaperMembership(
                workspace_id=job.workspace_id,
                paper_id=job.paper_id,
                state=job.state,
                added_at=self._clock(),
                attempts=job.attempts,
            )
            if job.state is not PaperState.QUEUED:
                # A redelivered job died mid-stage, and the pipeline restarts
                # from the beginning rather than resuming — which is only sound
                # because every step is idempotent (D52, D46). Saying so through
                # the state machine keeps the membership honest instead of
                # faking forward progress from a stage that never finished.
                membership.transition_to(PaperState.QUEUED)

            use_case = IngestPaperUseCase(
                papers=papers,
                sections=SqlSectionRepository(session, scope),
                chunks=SqlChunkRepository(session, scope),
                store=self._store,
                parser=self._parser,
                embedder=self._embedder,
                index=self._index,
                queue=queue,
                uow=SqlUnitOfWork(session),
                chunking=self._config.chunking,
                embed_batch_size=self._config.embed_batch_size,
            )
            outcome = await use_case.execute(
                job_id=job.id,
                workspace_id=job.workspace_id,
                paper_id=job.paper_id,
                membership=membership,
            )

            if outcome.succeeded:
                if not await queue.complete(job.id):
                    # The job row did not move. Since D92 the update carries a
                    # tenant predicate, so a miss means the row is gone or is
                    # not this scope's — and the consequence is specific: the
                    # lease expires, the job is re-claimed, and the paper is
                    # ingested again. Worth a line rather than a discarded bool.
                    log.warning(
                        "ingestion.completion_missed",
                        job_id=job.id,
                        paper_id=job.paper_id,
                        workspace_id=job.workspace_id,
                    )
                await session.commit()
                log.info(
                    "ingestion.indexed",
                    job_id=job.id,
                    paper_id=job.paper_id,
                    sections=outcome.sections,
                    chunks=outcome.chunks,
                    embedded=outcome.embedded,
                )
                return outcome

            try:
                await self._record_failure(job, outcome, queue=queue, papers=papers)
                await session.commit()
            except Exception as exc:
                # The classification survives the session that could not write
                # it. A DB-level failure earlier in the pipeline leaves this
                # session inside a failed transaction, so every later statement
                # raises — including the one recording WHY the paper failed. The
                # M1 log recorded the consequence as debt: the crash handler
                # would re-classify as TRANSIENT and the real reason
                # ("encrypted PDF", "no extractable text") was replaced by a
                # rollback error the user cannot act on. Carrying the outcome
                # on the exception lets `_abandon` write the original class and
                # detail on a fresh session instead.
                raise _FailureRecordingFailed(outcome, exc) from exc
            return outcome

    async def _abandon(self, job: QueuedJob, scope: TenantContext, exc: Exception) -> None:
        """Record an unexpected crash on a *fresh* session.

        The session the job was using may be inside a failed transaction — one
        `IntegrityError` and every later statement on it raises — so reusing it
        to write the failure would lose the failure too.

        **A crash that happened while RECORDING a classified failure keeps that
        classification** (see `_FailureRecordingFailed`). Without that, the most
        informative failures were the ones most likely to be rewritten: a paper
        the pipeline had correctly identified as UNSUPPORTED would come back as
        a retryable TRANSIENT carrying a rollback error.
        """
        if isinstance(exc, _FailureRecordingFailed):
            # The pipeline had already decided what went wrong; only the write
            # failed. Re-record THAT, not the write's error — otherwise a
            # database hiccup silently converts an UNSUPPORTED paper into a
            # retryable one and spends three deliveries rediscovering a failure
            # that was already known, while the user reads a rollback message
            # they cannot act on.
            outcome = exc.outcome
        else:
            outcome = IngestionOutcome(
                paper_id=job.paper_id,
                failure=IngestionFailure(
                    failure_class=FailureClass.TRANSIENT,
                    detail=f"{type(exc).__name__}: {exc}"[:500],
                ),
            )
        try:
            async with self._sessions() as session:
                await self._record_failure(
                    job,
                    outcome,
                    queue=PostgresQueue(session, scope),
                    papers=SqlPaperRepository(session, scope),
                )
                await session.commit()
        except Exception:
            # The database is unreachable or the row is gone. The lease still
            # expires, so the job is redelivered and the attempt counter still
            # bounds it — losing this write costs a retry, not the job.
            log.exception("ingestion.abandon_failed", job_id=job.id)

    async def _record_failure(
        self,
        job: QueuedJob,
        outcome: IngestionOutcome,
        *,
        queue: PostgresQueue,
        papers: SqlPaperRepository,
    ) -> None:
        """Apply the retry policy, then tell the membership what the queue decided.

        The queue owns the decision because only it knows how many deliveries
        have been spent (D48); the membership owns what the user sees. QUEUED
        back means "retrying" in §4.4.1's table, which is a different message
        from a terminal failure the user is invited to act on.
        """
        failure = outcome.failure
        if failure is None:  # pragma: no cover - `succeeded` is False, so there is one
            return
        landed = await queue.fail(
            job.id, failure_class=failure.failure_class, detail=failure.detail
        )
        await papers.set_membership_state(
            job.workspace_id,
            job.paper_id,
            landed,
            # Cleared on a retry: a membership that says "retrying" while
            # carrying last attempt's failure class reads as though it has
            # already given up.
            failure_class=None if landed is PaperState.QUEUED else failure.failure_class.value,
            failure_detail=None if landed is PaperState.QUEUED else failure.detail,
        )
        log.warning(
            "ingestion.failed",
            job_id=job.id,
            paper_id=job.paper_id,
            failure_class=failure.failure_class.value,
            attempts=job.attempts,
            landed=landed.value,
        )

    async def _owner_of(self, workspace_id: str) -> str | None:
        """Resolve the tenant this job belongs to — the one cross-tenant read.

        Soft-deleted workspaces are excluded for the same reason the scoped
        repositories exclude them: §5.6.4 marks DELETING and sweeps later, and
        ingesting into a workspace the user has deleted is work whose output
        nobody will ever see.
        """
        try:
            target = uuid.UUID(workspace_id)
        except ValueError:
            return None
        async with self._sessions() as session:
            owner = (
                await session.execute(
                    select(models.Workspace.owner_id).where(
                        models.Workspace.id == target,
                        models.Workspace.deleted_at.is_(None),
                    )
                )
            ).scalar_one_or_none()
        return str(owner) if owner is not None else None
