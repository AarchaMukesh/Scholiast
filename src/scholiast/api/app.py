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
from pathlib import Path
from typing import Any

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from scholiast.api.errors import register_error_handlers
from scholiast.api.routers import chat, health, papers, workspaces
from scholiast.application.ports.ml import Precision
from scholiast.config.settings import Settings, get_settings
from scholiast.infrastructure.db.engine import build_engine, build_session_factory
from scholiast.infrastructure.llm.chain import build_provider_chain
from scholiast.infrastructure.ml.embedder import LocalEmbedder
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

    register_error_handlers(app)
    app.include_router(health.router)
    app.include_router(workspaces.router)
    app.include_router(papers.router)
    app.include_router(papers.jobs_router)
    app.include_router(chat.router)

    if STATIC_DIR.is_dir():
        # Mounted last and at the root, so it cannot shadow an API route.
        app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="ui")
    return app
