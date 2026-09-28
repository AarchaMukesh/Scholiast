"""The Celery worker entrypoint (§5.7.4, D95, D96, D97).

Run with:

    celery -A scholiast.workers.tasks worker -Q ingest -c 1

**The task is a wake-up, not a work order.** Its argument is a job id, and it
does not use it to find work: it calls `IngestionWorker.run_once()`, which claims
whatever is due from Postgres under the lease. That is D95 made concrete, and it
buys three things at once. A duplicate delivery cannot double-process a paper,
because the claim is the serialisation point and the second claimer finds the row
gone. A redelivery after a crash needs no correlation between message and row,
because the lease already expired and the row is due again. And a message lost
entirely costs latency rather than work, because the sweep still finds the row.
The id travels for logs and tracing, which is why it is logged and not branched
on.

**One event loop per process, not one per task.** Celery's process model is
synchronous and would ordinarily wrap each task in `asyncio.run`, but the
pipeline holds an asyncpg pool, and a pool is bound to the loop that created it —
a fresh loop per task would mean a fresh pool per task, and a 419 MB encoder
reloaded alongside it. The process therefore keeps one loop and one set of
singletons for its lifetime, built on first use and reused.

**`acks_late` means this function must be safe to interrupt anywhere.** It is,
for the same reason the M1 loop was: every stage commits its own transition, and
a paper re-entering the pipeline resumes from the state its row records rather
than from the beginning (FR-I7).
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any

import structlog

from scholiast.application.ports.ml import Precision
from scholiast.config.settings import Settings, get_settings
from scholiast.infrastructure.db.engine import build_engine, build_session_factory
from scholiast.infrastructure.ml.embedder import LocalEmbedder
from scholiast.infrastructure.parsing.grobid import GrobidParser
from scholiast.infrastructure.queue.celery_app import INGEST_TASK, build_celery_app
from scholiast.infrastructure.storage.local import LocalObjectStore
from scholiast.infrastructure.vector.qdrant import QdrantVectorIndex
from scholiast.workers.ingestion import IngestionWorker, WorkerConfig

log = structlog.get_logger(__name__)

#: Celery discovers tasks on this module-level object.
app = build_celery_app(get_settings())


@dataclass
class _ProcessContext:
    """Everything a worker process builds once and keeps."""

    loop: asyncio.AbstractEventLoop
    worker: IngestionWorker
    #: Held here as well as inside the worker because the process has to make it
    #: ready before the first claim, and reaching into the worker's private
    #: attribute to do that would make this module depend on its internals.
    index: QdrantVectorIndex


_context: _ProcessContext | None = None


def _build(settings: Settings) -> tuple[IngestionWorker, QdrantVectorIndex]:
    """Construct the pipeline exactly as `scholiast.main` does in-process.

    Duplicated deliberately rather than shared with the composition root: that
    one wires a worker *into an API process* and reuses the API's singletons,
    while this one owns a process with no API in it. Merging them would mean one
    function that can only be read with both deployments in mind.
    """
    engine = build_engine(settings)
    index = QdrantVectorIndex(
        settings.qdrant_url,
        collection=settings.qdrant_collection,
        dimensions=settings.embedding_dimensions,
    )
    worker = IngestionWorker(
        sessions=build_session_factory(engine),
        store=LocalObjectStore(settings.object_store_path),
        parser=GrobidParser(settings.grobid_url),
        embedder=LocalEmbedder(
            settings.embedding_model,
            device=settings.embedding_device,
            precision=Precision(settings.embedding_precision),
            batch_size=settings.embedding_batch_size,
            gate=None,
        ),
        index=index,
        config=WorkerConfig(
            poll_seconds=settings.worker_poll_seconds,
            lease_seconds=settings.worker_lease_seconds,
            batch=settings.worker_batch_size,
            embed_batch_size=settings.embedding_batch_size,
        ),
    )
    return worker, index


def _process_context() -> _ProcessContext:
    """The per-process singletons, built on first task and kept.

    Not built at import time: importing this module is what `celery -A` does to
    *discover* tasks, and a module that opens a database pool and loads an
    encoder on import cannot be inspected without paying for both.
    """
    global _context
    if _context is None:
        settings = get_settings()
        loop = asyncio.new_event_loop()
        worker, index = _build(settings)
        loop.run_until_complete(index.ensure_ready())
        _context = _ProcessContext(loop=loop, worker=worker, index=index)
        log.info("worker.process_ready", queue=app.conf.task_default_queue)
    return _context


# Celery ships no type information for its decorator, so mypy sees an untyped
# callable wrapping a typed one. Narrow suppression rather than relaxing the
# rule for the package: the task body below stays fully checked.
@app.task(name=INGEST_TASK, bind=True, acks_late=True, ignore_result=True)  # type: ignore[untyped-decorator]
def ingest_paper(self: Any, job_id: str) -> int:  # noqa: ARG001 - bind gives retry context
    """Claim and run whatever is due. Returns how many jobs ran.

    The return value is for tests and Flower; nothing waits on it (there is no
    result backend, D95).
    """
    context = _process_context()
    ran = context.loop.run_until_complete(context.worker.run_once())
    log.info("worker.task_finished", dispatched_for=job_id, jobs_run=ran)
    return ran
