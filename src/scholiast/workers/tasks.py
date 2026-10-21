"""The Celery worker entrypoint (§5.7.4, D95, D96, D97).

Run with:

    celery -A scholiast.workers.tasks worker -Q ingest,backfill,maintenance -c 1
    celery -A scholiast.workers.tasks beat

The `-Q` list is `celery_app.WORKER_QUEUES`, and a test keeps the two equal: a
queue left off it is a queue whose messages nothing reads (D140).

**One beat, however many workers** (D100). Beat is a scheduler, not a consumer:
a second one does not share the schedule, it duplicates it, so every sweep would
run twice. Workers scale freely because the queue is what distributes their
work.

**In dev, one worker consumes every queue**, which means a long cascade still
occupies the slot an upload wants — the separation declared in `celery_app` is
structural until a deployment gives maintenance its own worker (M6). Said here
rather than left to be discovered, because the routing makes it look solved.
Between uploads and discovered papers the separation is real but is the
claim's (D139): whichever queue a wake-up came from, an upload runs first.

**The task is a wake-up, not a work order.** Its argument is a job id, and it
does not use it to find work: it calls `IngestionWorker.run_once()`, which claims
whatever is due from Postgres under the lease. That is D95 made concrete, and it
buys three things at once. A duplicate delivery cannot double-process a paper,
because the claim is the serialisation point and the second claimer finds the row
gone. A redelivery after a crash needs no correlation between message and row,
because the lease already expired and the row is due again. And a message lost
entirely costs latency rather than work, because beat's ingestion sweep finds
the row within `ingest_sweep_seconds` (D141). The id travels for logs and
tracing, which is why it is logged and not branched on.

**One wake-up drains everything due** (D141), so a sweep clears a backlog of
retries at once rather than one per pass.

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
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from scholiast.application.ports.ml import Precision
from scholiast.application.ports.tenancy import INDEX_CHECK_REASON, SystemContext
from scholiast.config.settings import Settings, get_settings
from scholiast.infrastructure.db.engine import build_engine, build_session_factory
from scholiast.infrastructure.db.repositories import SqlChunkRepository
from scholiast.infrastructure.fetch import build_pdf_fetcher
from scholiast.infrastructure.ml.embedder import LocalEmbedder
from scholiast.infrastructure.observability.logging import configure_logging
from scholiast.infrastructure.parsing.grobid import GrobidParser
from scholiast.infrastructure.queue.celery_app import (
    DELETE_TASK,
    INGEST_TASK,
    RECONCILE_TASK,
    build_celery_app,
)
from scholiast.infrastructure.storage.local import LocalObjectStore
from scholiast.infrastructure.vector.factory import build_vector_indexes
from scholiast.infrastructure.vector.mirrored import MirroredVectorIndex
from scholiast.infrastructure.vector.qdrant import QdrantVectorIndex
from scholiast.workers.ingestion import IngestionWorker, WorkerConfig
from scholiast.workers.maintenance import MaintenanceWorker

log = structlog.get_logger(__name__)

#: Celery discovers tasks on this module-level object.
app = build_celery_app(get_settings())

# FR-O1. At import, because `celery -A ... worker` imports this module and then
# starts logging from its own bootstrap before any task runs — configuring inside
# `_process_context` would leave the worker's startup lines unstructured. Cheap
# and idempotent, unlike the pool and the encoder, which is why those wait.
configure_logging(level=get_settings().log_level, json_format=get_settings().log_json)


@dataclass
class _ProcessContext:
    """Everything a worker process builds once and keeps."""

    loop: asyncio.AbstractEventLoop
    worker: IngestionWorker
    #: Shares the process's session factory and index with the ingestion worker.
    #: It holds no model and opens no parser, so a worker consuming only the
    #: maintenance queue still pays for the encoder — accepted for now because
    #: the dev topology runs one worker for everything (D100), and splitting the
    #: construction would buy nothing until the deployment splits too.
    maintenance: MaintenanceWorker
    #: Held here as well as inside the worker because the process has to make it
    #: ready before the first claim, and reaching into the worker's private
    #: attribute to do that would make this module depend on its internals.
    #: The write index: mirrored to the hybrid collection when one is set (D172).
    index: QdrantVectorIndex | MirroredVectorIndex


_context: _ProcessContext | None = None


def _build(
    settings: Settings,
) -> tuple[IngestionWorker, MaintenanceWorker, QdrantVectorIndex | MirroredVectorIndex]:
    """Construct the pipeline exactly as `scholiast.main` does in-process.

    Duplicated deliberately rather than shared with the composition root: that
    one wires a worker *into an API process* and reuses the API's singletons,
    while this one owns a process with no API in it. Merging them would mean one
    function that can only be read with both deployments in mind.
    """
    engine = build_engine(settings)
    sessions = build_session_factory(engine)
    store = LocalObjectStore(settings.object_store_path)
    # The same construction as the API's (D172): with a hybrid collection set,
    # every write this process makes reaches both collections, and a missing one
    # is created only while nothing is embedded (D174).
    indexes = build_vector_indexes(settings, embedded_chunks=lambda: _count_embedded(sessions))
    index = indexes.write
    worker = IngestionWorker(
        sessions=sessions,
        store=store,
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
        # Once per process, with the worker, so its pacing spans every job (D134).
        fetcher=build_pdf_fetcher(settings),
    )
    maintenance = MaintenanceWorker(
        sessions=sessions, index=index, store=store, hybrid=indexes.hybrid
    )
    return worker, maintenance, index


async def _count_embedded(sessions: async_sessionmaker[AsyncSession]) -> int:
    async with sessions() as session:
        return await SqlChunkRepository(
            session, SystemContext(reason=INDEX_CHECK_REASON)
        ).count_embedded()


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
        worker, maintenance, index = _build(settings)
        loop.run_until_complete(index.ensure_ready())
        _context = _ProcessContext(loop=loop, worker=worker, maintenance=maintenance, index=index)
        # The DEFAULT queue: the queues this process consumes are its `-Q` list,
        # which Celery holds and this module does not (D140).
        log.info("worker.process_ready", default_queue=app.conf.task_default_queue)
    return _context


# Celery ships no type information for its decorator, so mypy sees an untyped
# callable wrapping a typed one. Narrow suppression rather than relaxing the
# rule for the package: the task body below stays fully checked.
@app.task(name=INGEST_TASK, bind=True, acks_late=True, ignore_result=True)  # type: ignore[untyped-decorator]
def ingest_paper(self: Any, job_id: str | None = None) -> int:  # noqa: ARG001 - bind gives retry context
    """Claim and run everything due. Returns how many jobs ran.

    `job_id` is the job whose enqueue sent this wake-up, for the log; beat's
    sweep sends none (D141). The return value is for tests and Flower; nothing
    waits on it (there is no result backend, D95).
    """
    context = _process_context()
    ran = context.loop.run_until_complete(context.worker.drain())
    log.info("worker.task_finished", dispatched_for=job_id, jobs_run=ran)
    return ran


@app.task(name=DELETE_TASK, bind=True, acks_late=True, ignore_result=True)  # type: ignore[untyped-decorator]
def delete_workspace(self: Any, workspace_id: str) -> int:  # noqa: ARG001 - bind gives retry context
    """Sweep one deleted workspace out of all three stores (§5.6.4 step 2).

    **Unlike `ingest_paper`, the argument is load-bearing.** A wake-up can be
    vague because the ingestion worker claims whatever is due from Postgres;
    a cascade acts on the id it is given. That is why the use case refuses any
    workspace that is not already marked deleted — the id arrives over a broker
    and nothing else constrains it (D99).

    Returns how many papers were purged, for Flower and tests; nothing waits on
    it. A failure here is not retried by Celery and does not need to be: the
    mark is still in the database, so the next reconciliation pass finds it.
    """
    context = _process_context()
    result = context.loop.run_until_complete(context.maintenance.delete_workspace(workspace_id))
    return result.papers_purged


@app.task(name=RECONCILE_TASK, bind=True, acks_late=True, ignore_result=True)  # type: ignore[untyped-decorator]
def reconcile(self: Any) -> int:  # noqa: ARG001 - bind gives retry context
    """Converge the three stores onto the relational truth (§5.6.4 step 4).

    Scheduled by beat every `reconcile_interval_seconds` (D100). It takes no
    argument because it has no target: it asks the stores what is inconsistent
    rather than being told, which is what makes it a safety net for failures
    nobody recorded.

    Returns total residue removed. **Zero is the expected answer**, and anything
    else is a defect somewhere upstream that this merely cleaned up after.
    """
    context = _process_context()
    result = context.loop.run_until_complete(context.maintenance.reconcile())
    return result.papers_purged + result.vectors_deleted + result.blobs_deleted
