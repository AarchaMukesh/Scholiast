"""Dependency wiring — the one place adapters meet use cases (§5.3, D65).

This module is the composition root. Everything above it (routers) speaks DTOs
and use cases; everything below it (adapters) knows vendor names. Keeping the
knitting in one file is what makes the layering claim inspectable: if a router
imported `QdrantVectorIndex` directly, the contract would still pass — routers
are in `api`, which may import infrastructure — but the use case would have been
handed a concrete type and the swap-the-adapter claim would quietly stop being
true.

**`current_tenant` is §5.8 layer 1, and it is still the only place a principal is
constructed.** M1 left it returning a `TenantContext` for a seeded row and said
M3 would replace the inside and nothing else; M3 phase 1 did exactly that. Every
router depends on it and none of them verifies a token, so there is one function
to audit and one function to get wrong.

The layer-1 job is narrow on purpose: turn a bearer token into a principal, or
refuse. It never decides what that principal may *see* — that is layer 2's
repositories, which take the `TenantContext` this produces and apply the tenant
predicate to every query (D84, doc 5 §5.8).

**Long-lived singletons, per-request sessions.** The engine, the embedder, the
vector index and the provider chain are built once in the app's lifespan: loading
a 419 MB encoder per request would be absurd, and Qdrant's client does a server
version check on construction. The database session is per-request, because a
shared one would serialise every caller through one transaction.

**Two units of work, not one.** `get_session` rolls back on any exception;
`auth_session` commits on `AuthenticationFailed` as well, because a refused
authentication can still have written the security response to it (D86). Only
the two authentication repositories take the second one.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Annotated

from fastapi import Depends, Request
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy.ext.asyncio import AsyncSession

from scholiast.application.ports.accounts import AccountRepository, SessionRepository
from scholiast.application.ports.dispatch import JobDispatcher
from scholiast.application.ports.llm import LLMProvider
from scholiast.application.ports.ml import Embedder
from scholiast.application.ports.security import (
    AuthenticationFailed,
    PasswordHasher,
    TokenService,
)
from scholiast.application.ports.tenancy import (
    AUTHENTICATION_REASON,
    SystemContext,
    TenantContext,
)
from scholiast.application.ports.vector import VectorIndex
from scholiast.application.use_cases.add_paper import AddPaperUseCase
from scholiast.application.use_cases.answer_question import AnswerQuestionUseCase
from scholiast.application.use_cases.authentication import (
    EndSessionUseCase,
    RefreshSessionUseCase,
    RegisterUserUseCase,
    StartSessionUseCase,
)
from scholiast.application.use_cases.remove_paper import RemovePaperUseCase
from scholiast.config.settings import Settings
from scholiast.infrastructure.db import pending
from scholiast.infrastructure.db.accounts import SqlAccountRepository, SqlSessionRepository
from scholiast.infrastructure.db.repositories import (
    SqlChunkRepository,
    SqlConversationRepository,
    SqlPaperRepository,
    SqlSectionRepository,
    SqlWorkspaceRepository,
)
from scholiast.infrastructure.queue.postgres import PostgresQueue
from scholiast.infrastructure.storage.local import LocalObjectStore


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
            await _dispatch_committed_jobs(request, session)


async def _dispatch_committed_jobs(request: Request, session: AsyncSession) -> None:
    """Wake a worker for anything this transaction just committed (D95, D99).

    Two kinds travel this path and they are drained separately because they go
    to different tasks on different queues: accepted ingestion jobs, and
    workspaces the user just deleted. Both are already durable by the time this
    runs, and both have a backstop if the publish fails — a job's lease makes it
    claimable again, and a marked workspace is found by `list_deleting` on the
    next reconciliation pass.

    **After the commit, and deliberately outside it.** The job row is already
    durable, so this is a latency optimisation: if the broker is unreachable the
    job still runs, because its lease makes it claimable and the sweep finds it.
    That is the whole reason Celery can be the transport without giving up D47's
    single-transaction guarantee.

    The adapter reports failure by returning False rather than raising, and that
    result is **deliberately ignored here**: the request has already succeeded,
    turning a broker hiccup into a failed upload would be a worse answer than a
    slower one, and there is nothing useful this path can do about it. The cost
    is that a broker outage is currently invisible — it shows up as ingestion
    starting at the poll interval instead of immediately, with nothing saying
    why. `api` has no logger yet; phase 7 owns that, and this is a named place
    for it to land rather than an oversight.
    """
    jobs = pending.drain(session, pending.PENDING_DISPATCH)
    deletions = pending.drain(session, pending.PENDING_DELETION_DISPATCH)
    if not jobs and not deletions:
        return
    dispatcher: JobDispatcher = request.app.state.dispatcher
    for job_id in jobs:
        await dispatcher.dispatch(job_id)
    for workspace_id in deletions:
        await dispatcher.dispatch_deletion(workspace_id)


async def auth_session(request: Request) -> AsyncIterator[AsyncSession]:
    """The unit of work for the authentication tables — it commits on refusal too.

    **D86, and the phase-1 integration test is what found it.** D84's reuse
    detection revokes a whole refresh family and then raises, because the caller
    must still be refused. Under `get_session` that raise rolls the revocation
    back, so a stolen chain survives its own detection — measured: one live row
    where zero were expected, and the successor token still authenticated.

    A failed authentication is an expected outcome, not an error, and the writes
    it makes are the security *response* to it. The rule is confined to these two
    repositories rather than applied in `get_session`, so it cannot silently
    start preserving partial writes in some future route: the four auth routes
    reach `users` and `refresh_tokens` and nothing else, and the only write that
    can coexist with an `AuthenticationFailed` is the revocation.
    """
    factory = request.app.state.sessions
    async with factory() as session:
        try:
            yield session
        except AuthenticationFailed:
            await session.commit()
            raise
        except Exception:
            await session.rollback()
            raise
        else:
            await session.commit()


#: `auto_error=False` so a missing header reaches the code below rather than
#: producing FastAPI's own 403. Every rejection must leave here as one 401
#: (FR-A5), and a framework default that says "Not authenticated" with a
#: different status is a second shape callers can distinguish.
_bearer = HTTPBearer(auto_error=False, description="Access token from /v1/auth/login")


def current_tenant(
    request: Request,
    credentials: Annotated[HTTPAuthorizationCredentials | None, Depends(_bearer)],
) -> TenantContext:
    """Verify the bearer token and produce the principal (§5.8 layer 1).

    Signature verification only — no database read — which is what keeps replicas
    independent (D84). The consequence is stated where it is decided: an access
    token stays valid until it expires, so `access_token_minutes` IS the
    revocation latency.
    """
    tokens: TokenService = request.app.state.tokens
    if credentials is None or credentials.scheme.lower() != "bearer":
        raise AuthenticationFailed("no bearer credentials presented")
    claims = tokens.verify_access(credentials.credentials)
    return TenantContext(user_id=claims.subject)


def get_hasher(request: Request) -> PasswordHasher:
    hasher: PasswordHasher = request.app.state.hasher
    return hasher


def get_tokens(request: Request) -> TokenService:
    tokens: TokenService = request.app.state.tokens
    return tokens


def accounts(session: AuthSessionDep) -> SqlAccountRepository:
    """The account table, under the one sanctioned authentication bypass.

    `SystemContext` rather than `TenantContext` because this is what *produces* a
    principal — see `AUTHENTICATION_REASON`, which greps to exactly these two
    repositories and nothing else.
    """
    return SqlAccountRepository(session, SystemContext(AUTHENTICATION_REASON))


def sessions(session: AuthSessionDep) -> SqlSessionRepository:
    return SqlSessionRepository(session, SystemContext(AUTHENTICATION_REASON))


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
#: D86: the same session type, a different commit rule. Only the two
#: authentication repositories take it.
AuthSessionDep = Annotated[AsyncSession, Depends(auth_session)]
TenantDep = Annotated[TenantContext, Depends(current_tenant)]
IndexDep = Annotated[VectorIndex, Depends(get_index)]
EmbedderDep = Annotated[Embedder, Depends(get_embedder)]
ProvidersDep = Annotated[list[LLMProvider], Depends(get_providers)]
StoreDep = Annotated[LocalObjectStore, Depends(get_store)]
HasherDep = Annotated[PasswordHasher, Depends(get_hasher)]
TokensDep = Annotated[TokenService, Depends(get_tokens)]
AccountsDep = Annotated[AccountRepository, Depends(accounts)]
SessionsDep = Annotated[SessionRepository, Depends(sessions)]


def register_user_use_case(
    account_repo: AccountsDep, session_repo: SessionsDep, hasher: HasherDep, tokens: TokensDep
) -> RegisterUserUseCase:
    return RegisterUserUseCase(
        accounts=account_repo, sessions=session_repo, hasher=hasher, tokens=tokens
    )


def start_session_use_case(
    account_repo: AccountsDep, session_repo: SessionsDep, hasher: HasherDep, tokens: TokensDep
) -> StartSessionUseCase:
    return StartSessionUseCase(
        accounts=account_repo, sessions=session_repo, hasher=hasher, tokens=tokens
    )


def refresh_session_use_case(session_repo: SessionsDep, tokens: TokensDep) -> RefreshSessionUseCase:
    return RefreshSessionUseCase(sessions=session_repo, tokens=tokens)


def end_session_use_case(session_repo: SessionsDep, tokens: TokensDep) -> EndSessionUseCase:
    return EndSessionUseCase(sessions=session_repo, tokens=tokens)


RegisterUserDep = Annotated[RegisterUserUseCase, Depends(register_user_use_case)]
StartSessionDep = Annotated[StartSessionUseCase, Depends(start_session_use_case)]
RefreshSessionDep = Annotated[RefreshSessionUseCase, Depends(refresh_session_use_case)]
EndSessionDep = Annotated[EndSessionUseCase, Depends(end_session_use_case)]


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
        owner_id=scope.user_id,
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
