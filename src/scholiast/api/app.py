"""The application factory (§5.2, §5.5, D63).

**This layer does not know the worker exists**, and `import-linter` is why. The
layers contract makes `api` and `workers` independent siblings, so the obvious
implementation — the API's lifespan constructing an `IngestionWorker` — is a
contract violation. It is also a real design smell that the rule caught: an HTTP
layer that imports a background worker cannot be deployed without it. Composition
of the two lives in `scholiast.main`, above both (D63).

What stays here is everything the API itself needs: the engine, the session
factory, the object store, the embedder, the vector index and the provider chain.
`create_app` accepts a `lifespan` override, which is the seam `scholiast.main`
uses to wrap this one and add the worker.

**Expensive things are built once.** The engine, the embedder, the vector index
and the provider chain live on `app.state` for the process lifetime. Loading a
419 MB encoder per request would be absurd, and `AsyncQdrantClient` performs a
server version check when it connects. Database sessions are the exception and
are per-request, because a shared one would serialise every caller through a
single transaction.
"""

from __future__ import annotations

import contextlib
from collections.abc import AsyncIterator, Callable
from datetime import timedelta
from pathlib import Path
from typing import Any

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from scholiast.api.errors import register_error_handlers
from scholiast.api.middleware import ObservabilityMiddleware
from scholiast.api.routers import auth, chat, discovery, health, papers, workspaces
from scholiast.application.ports.ml import Precision
from scholiast.config.settings import Settings, get_settings
from scholiast.infrastructure.db.engine import build_engine, build_session_factory
from scholiast.infrastructure.discovery import tls_context
from scholiast.infrastructure.discovery.sources import build_discovery_sources
from scholiast.infrastructure.llm.chain import build_provider_chain
from scholiast.infrastructure.ml.embedder import LocalEmbedder
from scholiast.infrastructure.observability.logging import configure_logging
from scholiast.infrastructure.queue.celery_app import build_celery_app
from scholiast.infrastructure.queue.dispatch import CeleryDispatcher, NullDispatcher
from scholiast.infrastructure.security.passwords import Argon2idHasher
from scholiast.infrastructure.security.ratelimit import FixedWindowLimiter
from scholiast.infrastructure.security.tokens import JwtTokenService
from scholiast.infrastructure.storage.local import LocalObjectStore
from scholiast.infrastructure.vector.qdrant import QdrantVectorIndex

STATIC_DIR = Path(__file__).parent / "static"


@contextlib.asynccontextmanager
async def api_lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Build the API's singletons and tear them down in order."""
    settings: Settings = app.state.settings
    engine = build_engine(settings)
    app.state.engine = engine
    app.state.sessions = build_session_factory(engine)
    app.state.store = LocalObjectStore(settings.object_store_path)
    # Cheap, stateless and shared: the hasher holds only its cost parameters, and
    # the token service only its key and lifetimes. Built here anyway so the
    # secret is validated at startup — a deployment with a too-short key should
    # fail to boot, not fail at the first login (D84).
    app.state.hasher = Argon2idHasher()
    app.state.tokens = JwtTokenService(
        secret=settings.jwt_secret,
        issuer=settings.jwt_issuer,
        access_ttl=timedelta(minutes=settings.access_token_minutes),
        refresh_ttl=timedelta(days=settings.refresh_token_days),
    )
    # D95/D96: the API wakes a worker by task NAME and never imports one. With
    # Celery off, the in-process worker is already polling the same database, so
    # the null dispatcher keeps the call site free of a topology branch.
    if settings.celery_enabled:
        app.state.celery = build_celery_app(settings)
        app.state.dispatcher = CeleryDispatcher(app.state.celery)
    else:
        app.state.celery = None
        app.state.dispatcher = NullDispatcher()

    # D97 (amended): no production embedder is handed a gate, here or in the
    # Celery worker, because none ever was — `eval/cli.py` is the only caller
    # that builds one. `RedisGpuGate` exists and is tested for the change that
    # switches gating on, which is its own change precisely because a gated
    # embedder re-batches and perturbs fp16 vectors (D74). Constructing one here
    # that nothing consumes would advertise a capability that is not connected.

    app.state.embedder = LocalEmbedder(
        settings.embedding_model,
        device=settings.embedding_device,
        precision=Precision(settings.embedding_precision),
        batch_size=settings.embedding_batch_size,
    )
    index = QdrantVectorIndex(
        settings.qdrant_url,
        collection=settings.qdrant_collection,
        dimensions=settings.embedding_dimensions,
    )
    app.state.index = index
    app.state.providers = build_provider_chain(settings)
    # Build discovery's shared TLS context now, so the first search after a restart
    # does not spend 183 ms of blocked event loop on it (D122).
    tls_context()

    # Creates the collection and its payload index, or refuses to start if an
    # existing collection was built for a different embedder (D46, and the
    # VectorDimensionMismatch phase 4 added). Failing here beats failing on
    # every upsert deep inside ingestion, far from the cause.
    await index.ensure_ready()

    try:
        yield
    finally:
        await index.close()
        await engine.dispose()


def create_app(
    settings: Settings | None = None,
    *,
    lifespan: Callable[[FastAPI], Any] | None = None,
) -> FastAPI:
    """Build the application.

    A factory rather than a module-level instance so tests can construct one
    against throwaway services without importing a process-wide singleton that
    already connected to the developer's database.

    `lifespan` overrides the default so a composition root can wrap it — which is
    how `scholiast.main` adds the ingestion worker without this layer importing
    it.
    """
    app = FastAPI(
        title="Scholiast",
        version="0.1.0",
        # NFR-M6: the version lives in the path, and the OpenAPI document that
        # NFR-DOC2 asks for comes free with it.
        description=(
            "Retrieval-and-reasoning over academic literature, with measured "
            "retrieval quality. M1 walking skeleton."
        ),
        lifespan=lifespan or api_lifespan,
    )
    app.state.settings = settings or get_settings()
    # Defaulted HERE, not in the lifespan, because `create_app` takes a lifespan
    # override (that is how `scholiast.main` adds the worker and how the tests
    # build a Postgres-only app). A dependency that every override has to
    # remember to set is one that some override will forget — and this one is
    # reached on the upload path, so forgetting it turns an accepted paper into
    # a 500. The real lifespan replaces it when Celery is on.
    app.state.dispatcher = NullDispatcher()
    app.state.celery = None
    # NFR-SEC7. Here for the same reason as the dispatcher: a lifespan override
    # that forgot it would 500 on the first limited route. One per app, which in
    # production is one per process (D112).
    app.state.rate_limiter = FixedWindowLimiter()
    # Discovery's sources, breakers and arXiv pacer (D122, D123). Here and not in
    # the lifespan for the limiter's reason, and because they are state that must
    # outlive a request: a pacer rebuilt per request would pace nothing. Building
    # them opens no connection, so a test app pays nothing for them.
    app.state.discovery_sources = build_discovery_sources(app.state.settings)

    # FR-O1. Configured here rather than in the lifespan for the same reason the
    # dispatcher is: every lifespan override would otherwise have to remember it,
    # and the one that forgot would produce an app whose logs are unstructured
    # with nothing failing to say so. Idempotent, so the worker process calling
    # it too is not a conflict.
    configure_logging(level=app.state.settings.log_level, json_format=app.state.settings.log_json)
    # Outermost, so it sees the status and duration of everything including the
    # error handlers below — a middleware added after them would time the app
    # and miss the response they produce.
    app.add_middleware(ObservabilityMiddleware)

    register_error_handlers(app)
    app.include_router(auth.router)
    app.include_router(health.router)
    app.include_router(workspaces.router)
    app.include_router(papers.router)
    app.include_router(papers.jobs_router)
    app.include_router(chat.router)
    app.include_router(discovery.router)

    if STATIC_DIR.is_dir():
        # Mounted last and at the root, so it cannot shadow an API route.
        app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="ui")
    return app
