"""Ingestion queue port (§5.4.2, §5.7.4, D47, D48).

The M1 adapter stores jobs in Postgres and claims them with `SKIP LOCKED`, which
makes one of §5.4.2's rules disappear rather than satisfy it: "the job row is
committed *before* the enqueue" exists because a crash between two stores loses
work, and with the queue *in* the source of truth the paper row, the membership
row and the job row commit together. There is no window.

What the port still has to express, because M3 swaps the transport for Celery:

  lease        a claim is a lease, not a removal. A worker that dies mid-job
               does not take the job with it — the row becomes claimable again
               when `visible_at` passes (§5.7.4's visibility timeout).

  attempts     retry is bounded and the count is durable, because the decision
               "retry or dead-letter" has to survive the process that was
               counting (D48).

  stage        `state` carries the pipeline stage rather than a generic
               "in-flight", so an operator and `GET /v1/jobs/{id}` can see where
               a paper actually is (FR-I3, FR-O5).

Nothing here mentions SQL, and nothing here takes a session: the adapter is
constructed with one, exactly as the repositories are, which is what lets a use
case commit a job alongside the rows it belongs to.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from scholiast.domain.errors import FailureClass
from scholiast.domain.workspace import PaperState

#: D48. Three attempts on a five-second base survives a GROBID restart (~30 s)
#: and gives up on a genuinely broken paper inside a minute. A judgement call,
#: not a measurement — stated here so it is one number rather than three.
DEFAULT_MAX_ATTEMPTS = 3
DEFAULT_RETRY_BASE_SECONDS = 5.0
#: Jitter spreads a thundering herd of simultaneous retries. ±20% is enough to
#: decorrelate the small number of jobs M1 can have in flight.
RETRY_JITTER_RATIO = 0.2

#: How long a claim is held before the job is redelivered. Must exceed the
#: slowest stage by a wide margin: GROBID is ~3 s/paper and embedding a large
#: paper is seconds more, so a five-minute lease means only a genuinely dead
#: worker loses its claim — while a live one refreshes it at every stage.
DEFAULT_LEASE_SECONDS = 300.0


@dataclass(frozen=True, slots=True)
class QueuedJob:
    """A claimed unit of ingestion work.

    Carries `attempts` because the retry decision belongs to the caller that
    knows *how* the attempt failed — the queue knows only that it did.
    """

    id: str
    workspace_id: str
    paper_id: str
    state: PaperState
    attempts: int


@runtime_checkable
class Queue(Protocol):
    """Durable, at-least-once ingestion work."""

    async def enqueue(self, *, workspace_id: str, paper_id: str) -> str:
        """Record a job and return its id.

        At-least-once delivery plus an idempotent consumer (FR-I7) is the
        contract; exactly-once is not offered because it cannot be honoured
        across a process boundary, and pretending otherwise is how work gets
        silently dropped.
        """
        ...

    async def claim(
        self, *, limit: int = 1, lease_seconds: float = DEFAULT_LEASE_SECONDS
    ) -> list[QueuedJob]:
        """Take up to `limit` due jobs, leasing them for `lease_seconds`.

        Concurrent claimers must never receive the same job — that is the
        `SKIP LOCKED` half of the adapter, and it is what makes a second worker
        a throughput change rather than a correctness one.
        """
        ...

    async def heartbeat(
        self, job_id: str, *, state: PaperState, lease_seconds: float = DEFAULT_LEASE_SECONDS
    ) -> bool:
        """Record a stage transition and extend the lease.

        One call, because the two are the same event: a worker that has moved to
        the next stage is demonstrably alive, and splitting them invites a job
        whose stage is current and whose lease is not.
        """
        ...

    async def complete(self, job_id: str) -> bool:
        """Mark the job INDEXED. Terminal — never claimed again."""
        ...

    async def fail(
        self,
        job_id: str,
        *,
        failure_class: FailureClass,
        detail: str,
        max_attempts: int = DEFAULT_MAX_ATTEMPTS,
    ) -> PaperState:
        """Apply the retry policy and report where the job landed (D48).

        Returns `QUEUED` when the job will be retried — the caller needs to know
        that in order to leave the membership showing "retrying" rather than a
        terminal failure the user cannot act on (§4.4.1). Returns `FAILED` on a
        retryable class whose attempts are exhausted, and `UNSUPPORTED` for a
        class that retrying cannot help.
        """
        ...
