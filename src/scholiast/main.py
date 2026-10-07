"""The composition root — the one command M1's acceptance criterion asks for.

    uvicorn scholiast.main:app

**This module exists because `import-linter` refused the obvious alternative**,
and the rule was right. The layers contract makes `scholiast.api` and
`scholiast.workers` independent siblings, so an API lifespan that constructed an
`IngestionWorker` was a contract violation — and a real design smell underneath
it: an HTTP layer that imports a background worker cannot be deployed without
one, which is precisely the coupling M3 has to undo when Celery arrives.

So composition lives above both. `scholiast.main` is in no layer (the contract is
`exhaustive = false`), which is the standard place for a composition root: it is
allowed to know about everything because it is what wires everything, and it
contains no logic of its own to be tested.

**Why the worker runs here at all** (D63): doc 8 specifies an in-process queue in
M1, and the acceptance criterion is `docker compose up` plus *one* command — so
that command has to start both halves. `worker_in_process` turns it off, which is
how M3 splits them: the flag goes off, Celery runs the worker elsewhere, and this
module shrinks back to `create_app()`. The accepted cost is that restarting the
API interrupts in-flight ingestion, which is safe only because D47's lease
redelivers the job and D52/D46 make re-running it converge on the same rows.
"""

from __future__ import annotations

import asyncio
import contextlib
from collections.abc import AsyncIterator

import structlog
from fastapi import FastAPI

from scholiast.api.app import api_lifespan, create_app
from scholiast.config.settings import Settings, get_settings
from scholiast.infrastructure.fetch import build_pdf_fetcher
from scholiast.infrastructure.parsing.grobid import GrobidParser
from scholiast.workers.ingestion import IngestionWorker, WorkerConfig

log = structlog.get_logger(__name__)


@contextlib.asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """The API's lifespan, wrapped to add the ingestion worker.

    Nested rather than reimplemented: the worker needs the session factory, the
    object store, the embedder and the vector index that `api_lifespan` builds,
    and duplicating that construction would give the two halves separate
    instances of a 419 MB encoder.
    """
    async with api_lifespan(app):
        settings: Settings = app.state.settings
        if not settings.worker_in_process:
            log.info("worker.disabled", reason="worker_in_process is off")
            yield
            return

        worker = IngestionWorker(
            sessions=app.state.sessions,
            store=app.state.store,
            parser=GrobidParser(settings.grobid_url),
            embedder=app.state.embedder,
            index=app.state.index,
            config=WorkerConfig(
                poll_seconds=settings.worker_poll_seconds,
                lease_seconds=settings.worker_lease_seconds,
                batch=settings.worker_batch_size,
                embed_batch_size=settings.embedding_batch_size,
            ),
            # Built once with the worker, so its pacing spans every job (D134).
            fetcher=build_pdf_fetcher(settings),
        )
        task = asyncio.create_task(worker.run_forever())
        log.info("worker.started", poll_seconds=settings.worker_poll_seconds)
        try:
            yield
        finally:
            worker.stop()
            # The loop wakes on the stop event rather than after a poll interval,
            # so this returns promptly. The timeout is a backstop against a job
            # that is mid-stage, not the expected path — and letting it expire
            # is safe, because an abandoned job comes back on its lease.
            with contextlib.suppress(TimeoutError, asyncio.CancelledError):
                await asyncio.wait_for(task, timeout=30)
            log.info("worker.stopped")


def build() -> FastAPI:
    """The application, with the worker attached."""
    return create_app(get_settings(), lifespan=lifespan)


app = build()
