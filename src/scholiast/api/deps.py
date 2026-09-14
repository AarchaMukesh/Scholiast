"""Dependency wiring — the one place adapters meet use cases (§5.3, D65).

This module is the composition root. Everything above it (routers) speaks DTOs
and use cases; everything below it (adapters) knows vendor names. Keeping the
knitting in one file is what makes the layering claim inspectable: if a router
imported `QdrantVectorIndex` directly, the contract would still pass — routers
are in `api`, which may import infrastructure — but the use case would have been
handed a concrete type and the swap-the-adapter claim would quietly stop being
true.

**`current_tenant` is the entire auth surface in M1.** Doc 8's trivial adapter is
"single seeded user", and this is it: one function returning a `TenantContext`
for the row migration 0003 seeds. It is deliberately the *only* place a principal
is constructed, so M3 replaces JWT verification here and nowhere else — which is
§5.8 layer 1 in the shape it will keep, with a stub principal inside it.

**Long-lived singletons, per-request sessions.** The engine, the embedder, the
vector index and the provider chain are built once in the app's lifespan: loading
a 419 MB encoder per request would be absurd, and Qdrant's client does a server
version check on construction. The database session is per-request, because a
shared one would serialise every caller through one transaction.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Annotated

from fastapi import Depends, Request
from sqlalchemy.ext.asyncio import AsyncSession

from scholiast.application.ports.llm import LLMProvider
from scholiast.application.ports.ml import Embedder
from scholiast.application.ports.tenancy import TenantContext
from scholiast.application.ports.vector import VectorIndex
from scholiast.application.use_cases.add_paper import AddPaperUseCase
from scholiast.application.use_cases.answer_question import AnswerQuestionUseCase
from scholiast.application.use_cases.remove_paper import RemovePaperUseCase
from scholiast.config.settings import Settings
from scholiast.infrastructure.db.repositories import (
    SqlChunkRepository,
    SqlConversationRepository,
    SqlPaperRepository,
    SqlSectionRepository,
    SqlWorkspaceRepository,
)
from scholiast.infrastructure.queue.postgres import PostgresQueue
from scholiast.infrastructure.storage.local import LocalObjectStore

#: The user migration 0003 seeds. Fixed so the UI and the tests can name the
#: principal without a lookup; M3 replaces this whole mechanism with real auth,
#: and the recorded debt says `users.password_hash` must become NOT NULL then.
M1_USER_ID = "00000000-0000-4000-8000-000000000001"


async def get_session(request: Request) -> AsyncIterator[AsyncSession]:
    """One unit of work per request: commit on success, roll back on failure.

    The commit lives here rather than in each route so no handler can forget it
    — and so `AddPaperUseCase`'s single-transaction guarantee (the paper, the
    membership and the job row committing together, D47) holds by construction
    rather than by every caller remembering to.
    """
    factory = request.app.state.sessions
    async with factory() as session:
        try:
            yield session
        except Exception:
            await session.rollback()
            raise
        else:
            await session.commit()


def current_tenant() -> TenantContext:
    """The authenticated principal. In M1 there is exactly one (doc 8)."""
    return TenantContext(user_id=M1_USER_ID)


def get_settings_dep(request: Request) -> Settings:
    settings: Settings = request.app.state.settings
    return settings


def get_index(request: Request) -> VectorIndex:
    index: VectorIndex = request.app.state.index
    return index


def get_embedder(request: Request) -> Embedder:
    embedder: Embedder = request.app.state.embedder
    return embedder


def get_providers(request: Request) -> list[LLMProvider]:
    providers: list[LLMProvider] = request.app.state.providers
    return providers


def get_store(request: Request) -> LocalObjectStore:
    store: LocalObjectStore = request.app.state.store
    return store


SessionDep = Annotated[AsyncSession, Depends(get_session)]
TenantDep = Annotated[TenantContext, Depends(current_tenant)]
IndexDep = Annotated[VectorIndex, Depends(get_index)]
EmbedderDep = Annotated[Embedder, Depends(get_embedder)]
ProvidersDep = Annotated[list[LLMProvider], Depends(get_providers)]
StoreDep = Annotated[LocalObjectStore, Depends(get_store)]


def workspaces(session: SessionDep, scope: TenantDep) -> SqlWorkspaceRepository:
    return SqlWorkspaceRepository(session, scope)


def papers(session: SessionDep, scope: TenantDep) -> SqlPaperRepository:
    return SqlPaperRepository(session, scope)


def sections(session: SessionDep, scope: TenantDep) -> SqlSectionRepository:
    return SqlSectionRepository(session, scope)


def conversations(session: SessionDep, scope: TenantDep) -> SqlConversationRepository:
    return SqlConversationRepository(session, scope)


WorkspacesDep = Annotated[SqlWorkspaceRepository, Depends(workspaces)]
PapersDep = Annotated[SqlPaperRepository, Depends(papers)]
SectionsDep = Annotated[SqlSectionRepository, Depends(sections)]
ConversationsDep = Annotated[SqlConversationRepository, Depends(conversations)]


def add_paper_use_case(
    session: SessionDep,
    scope: TenantDep,
    store: StoreDep,
    workspace_repo: WorkspacesDep,
    paper_repo: PapersDep,
) -> AddPaperUseCase:
    return AddPaperUseCase(
        workspaces=workspace_repo,
        papers=paper_repo,
        chunks=SqlChunkRepository(session, scope),
        store=store,
        # Same session as the repositories above — that is what makes the paper,
        # the membership and the job row one transaction (D47).
        queue=PostgresQueue(session, scope),
        owner_id=M1_USER_ID,
    )


def remove_paper_use_case(
    store: StoreDep,
    index: IndexDep,
    workspace_repo: WorkspacesDep,
    paper_repo: PapersDep,
) -> RemovePaperUseCase:
    return RemovePaperUseCase(
        workspaces=workspace_repo, papers=paper_repo, index=index, store=store
    )


def answer_use_case(
    session: SessionDep,
    scope: TenantDep,
    embedder: EmbedderDep,
    index: IndexDep,
    providers: ProvidersDep,
    workspace_repo: WorkspacesDep,
    paper_repo: PapersDep,
    conversation_repo: ConversationsDep,
) -> AnswerQuestionUseCase:
    return AnswerQuestionUseCase(
        workspaces=workspace_repo,
        papers=paper_repo,
        chunks=SqlChunkRepository(session, scope),
        conversations=conversation_repo,
        embedder=embedder,
        index=index,
        providers=providers,
    )


AddPaperDep = Annotated[AddPaperUseCase, Depends(add_paper_use_case)]
RemovePaperDep = Annotated[RemovePaperUseCase, Depends(remove_paper_use_case)]
AnswerDep = Annotated[AnswerQuestionUseCase, Depends(answer_use_case)]
