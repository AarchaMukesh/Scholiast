"""Celery dispatch, addressed by name (D95, D96).

`send_task` rather than `ingest_paper.delay(...)`, and the difference is
architectural rather than stylistic. `import-linter` makes `scholiast.api` and
`scholiast.workers` independent siblings, so the API cannot import the task — and
the rule is catching a deployment fact, exactly as it did in D63: importing the
task module would drag the pipeline, the parser and the ML stack into a process
that never runs them. Celery addresses tasks by name over the wire, so the API
needs the name and a broker connection, not the code.

**Failures are swallowed on purpose.** By the time this runs, the job row is
committed; the paper is accepted and will be ingested whether or not Redis is
reachable, because the lease makes an unclaimed job claimable again and the sweep
finds it. Raising here would turn a latency problem into a failed upload, and
would do it *after* the transaction the user's request already succeeded in.
"""

from __future__ import annotations

import asyncio

from celery import Celery

from scholiast.infrastructure.queue.celery_app import (
    DELETE_TASK,
    INGEST_QUEUE,
    INGEST_TASK,
    MAINTENANCE_QUEUE,
)


class CeleryDispatcher:
    """`JobDispatcher` over Celery."""

    def __init__(
        self,
        app: Celery,
        *,
        queue: str = INGEST_QUEUE,
        maintenance_queue: str = MAINTENANCE_QUEUE,
    ) -> None:
        self._app = app
        self._queue = queue
        self._maintenance_queue = maintenance_queue

    async def dispatch(self, job_id: str) -> bool:
        """Publish a wake-up for one committed job.

        `send_task` opens a socket, so it runs off the event loop: the request
        that triggered it has already done its database work and a few
        milliseconds of broker I/O on the loop would be paid by every other
        in-flight request.
        """
        try:
            await asyncio.to_thread(self._send, job_id)
        except Exception:
            # Deliberately broad. Any failure to reach the broker — connection
            # refused, auth, a serialisation bug in kombu — has the same
            # consequence and the same remedy: the job waits for the sweep.
            return False
        return True

    async def dispatch_deletion(self, workspace_id: str) -> bool:
        """Publish a wake-up for one committed workspace deletion (D99).

        Same best-effort contract as `dispatch`, resting on a different
        backstop: an undispatched job is found by its lease, an undispatched
        deletion by `list_deleting` on the next reconciliation pass. The user
        already has their 202 and the workspace already stopped serving reads,
        so a broker failure here delays the sweep rather than losing it.
        """
        try:
            await asyncio.to_thread(self._send_deletion, workspace_id)
        except Exception:
            return False
        return True

    def _send(self, job_id: str) -> None:
        self._app.send_task(INGEST_TASK, args=[job_id], queue=self._queue)

    def _send_deletion(self, workspace_id: str) -> None:
        self._app.send_task(DELETE_TASK, args=[workspace_id], queue=self._maintenance_queue)


class NullDispatcher:
    """Dispatches nothing, for the single-process deployment (D63).

    With `worker_in_process` true the worker is already polling the same
    database, so a broker round trip would buy nothing. This exists so that
    configuration is the only difference between the two topologies — the call
    site does not branch on which one it is talking to.
    """

    async def dispatch(self, job_id: str) -> bool:  # noqa: ARG002 - the point is to do nothing
        return False

    async def dispatch_deletion(self, workspace_id: str) -> bool:  # noqa: ARG002 - as above
        """Nothing to wake. The reconciliation sweep finds the marked row.

        Unlike the ingestion case there is no in-process poller watching for
        deletions, so with no broker the cascade waits for the next sweep rather
        than starting immediately. That is latency, not loss — the mark is
        committed and `list_deleting` is what finds it (D99).
        """
        return False
