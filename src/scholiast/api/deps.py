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

**Both commit before the response is sent** (D166): `SessionDep` and
`AuthSessionDep` are function-scoped, so a client that receives a 2xx can read
what it wrote on its very next request. The one exception is the streamed
answer, whose turns are written while the body streams; it has its own
request-scoped session, `stream_session`, and its own repositories built on it.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Callable
from typing import Annotated

import structlog
from fastapi import BackgroundTasks, Depends, Request
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy.ext.asyncio import AsyncSession

from scholiast.api.errors import RateLimited
from scholiast.api.schemas import AddDiscoveredPapers
from scholiast.application.ports.accounts import AccountRepository, SessionRepository
from scholiast.application.ports.discovery import VectorCache
from scholiast.application.ports.dispatch import JobDispatcher
from scholiast.application.ports.llm import LLMProvider
from scholiast.application.ports.ml import Embedder
from scholiast.application.ports.queue import JobPriority
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
from scholiast.application.use_cases.add_discovered import AddDiscoveredPapersUseCase
from scholiast.application.use_cases.add_paper import AddPaperUseCase
from scholiast.application.use_cases.answer_question import AnswerQuestionUseCase
from scholiast.application.use_cases.authentication import (
    EndSessionUseCase,
    RefreshSessionUseCase,
    RegisterUserUseCase,
    StartSessionUseCase,
)
from scholiast.application.use_cases.discover_papers import (
    ConfiguredSource,
    DiscoverPapersUseCase,
)
from scholiast.application.use_cases.rank_works import DenseRelevance
from scholiast.application.use_cases.remove_paper import RemovePaperUseCase
from scholiast.config.settings import Settings
from scholiast.domain.fetch_policy import host_allowlist
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
from scholiast.infrastructure.security.ratelimit import FixedWindowLimiter, Limit
from scholiast.infrastructure.storage.local import LocalObjectStore

log = structlog.get_logger(__name__)


async def get_session(
    request: Request, background_tasks: BackgroundTasks
) -> AsyncIterator[AsyncSession]:
    """One unit of work per request: commit on success, roll back on failure.

    The commit lives here rather than in each route so no handler can forget it
    — and so `AddPaperUseCase`'s single-transaction guarantee (the paper, the
    membership and the job row committing together, D47) holds by construction
    rather than by every caller remembering to.

    **The commit runs before the response is sent** (D166), because `SessionDep`
    declares this function-scoped: FastAPI closes it when the handler returns,
    not after the response, which is when a request-scoped dependency closes.
    Committed after the response, a 2xx could reach a client before its write
    did, and the client's next request could read the state before it. Seen: the
    workspace counter that stayed at "0/1 ready" after a paper was added. Possible
    by the same race: a workspace the first-login flow creates twice, a login
    refused just after registering, and a commit that fails after a success was
    already sent.

    **Waking a worker still happens after the response**, as a background task:
    with a broker that is down, `send_task` waits out its publish retries, and
    that wait belongs to no user's request (D95). The pending ids are taken here,
    while the session is open; only the publishing is deferred.
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
            jobs, deletions = _take_committed(session)
            if jobs or deletions:
                background_tasks.add_task(_publish, request.app.state.dispatcher, jobs, deletions)


async def stream_session(request: Request) -> AsyncIterator[AsyncSession]:
    """The unit of work for a streamed response: it lives until the stream ends.

    The one deliberate exception to D166. The answer route returns a
    `StreamingResponse` and only afterwards, while the body streams, does the use
    case record the question and the answer (`AnswerQuestionUseCase._persist`); a
    session closed when the handler returned would be gone by then. So this one is
    request-scoped, and the streaming route builds every repository it uses from
    it, so one request still holds exactly one session. Its commit follows the
    stream, as every commit did before D166; what it commits is the record of an
    answer the client has already read in full.
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


def _take_committed(
    session: AsyncSession,
) -> tuple[list[tuple[str, JobPriority]], list[str]]:
    """The jobs and deletions this transaction committed, taken off the session."""
    jobs = [
        (job_id, JobPriority.INTERACTIVE)
        for job_id in pending.drain(session, pending.PENDING_DISPATCH)
    ] + [
        (job_id, JobPriority.BACKFILL)
        for job_id in pending.drain(session, pending.PENDING_BACKFILL_DISPATCH)
    ]
    return jobs, pending.drain(session, pending.PENDING_DELETION_DISPATCH)


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

    The adapter reports failure by returning False rather than raising, and the
    request still succeeds: turning a broker hiccup into a failed upload would be
    a worse answer than a slower one, and the lease already covers the work.

    **What phase 7 changed is that the failure is no longer invisible** (FR-O1).
    Phases 3 and 5 both left this comment saying `api` had no logger and a broker
    outage therefore showed up only as ingestion starting at the poll interval
    with nothing saying why. It now logs at warning, with the correlation id the
    middleware bound, so the slow path is attributable to the outage that caused
    it. The result is still not raised — logging it is the useful thing available
    here, and failing the request is not.
    """
    jobs, deletions = _take_committed(session)
    await _publish(request.app.state.dispatcher, jobs, deletions)


async def _publish(
    dispatcher: JobDispatcher,
    jobs: list[tuple[str, JobPriority]],
    deletions: list[str],
) -> None:
    """Publish one wake-up per committed job and deletion; log, never raise."""
    for job_id, priority in jobs:
        published = await dispatcher.dispatch(job_id, priority=priority)
        if dispatcher.publishes and not published:
            log.warning(
                "dispatch.not_published", job_id=job_id, kind="ingestion", priority=priority.label
            )
    for workspace_id in deletions:
        published = await dispatcher.dispatch_deletion(workspace_id)
        if dispatcher.publishes and not published:
            log.warning("dispatch.not_published", workspace_id=workspace_id, kind="deletion")


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


#: Function-scoped (D166): committed when the handler returns, before the response.
SessionDep = Annotated[AsyncSession, Depends(get_session, scope="function")]
#: D86: the same session type, a different commit rule. Only the two
#: authentication repositories take it. Function-scoped for the same reason (D166):
#: a client that registers and then logs in must find its account committed.
AuthSessionDep = Annotated[AsyncSession, Depends(auth_session, scope="function")]
#: Request-scoped on purpose: only the streamed answer takes it (D166).
StreamSessionDep = Annotated[AsyncSession, Depends(stream_session)]
TenantDep = Annotated[TenantContext, Depends(current_tenant)]


# --- rate limiting (NFR-SEC7, D112, D113) ---------------------------------
#
# A dependency rather than middleware, because the key needs the authenticated
# principal and middleware runs before routing and before auth (D112). The known
# cost is that every protected route has to remember it — the same trap phases 5
# and 7 both hit — so the completeness test enumerates the routes that must carry
# one and fails if a new expensive endpoint arrives without it.

#: The limiter lives on `app.state`, defaulted in `create_app` alongside the
#: dispatcher and for the same reason. It was module-level first, and that was
#: wrong twice over: state shared across every app in a process makes one test's
#: requests count against another's, so test order would change results — and in
#: production there is exactly one app per process anyway, which makes app-scoped
#: and process-scoped the same thing. The isolation is free.


def _peer_key(request: Request) -> str:
    """The socket peer. **`X-Forwarded-For` is deliberately not read** (D113).

    Nothing sits in front of this app, so honouring the header would let any
    caller mint a fresh bucket per forged value — a control an attacker opts out
    of with one header — and would let them attribute their traffic to somebody
    else's address. Under a future proxy this fails CLOSED: anonymous callers
    share the proxy's bucket, which is too strict and says so, rather than
    bypassable and silent.
    """
    client = request.client
    return f"ip:{client.host}" if client is not None else "ip:unknown"


def _enforce(request: Request, *, bucket: str, allowance: int, key: str, cost: int = 1) -> None:
    settings: Settings = request.app.state.settings
    if not settings.rate_limit_enabled:
        return
    limiter: FixedWindowLimiter = request.app.state.rate_limiter
    decision = limiter.check(
        key=key, bucket=bucket, limit=Limit(requests=allowance, window_seconds=60.0), cost=cost
    )
    if not decision.allowed:
        # Warning, not info: on a healthy system nobody hits these, so every line
        # is either an attack or a limit set too low. Both want looking at.
        log.warning("ratelimit.refused", bucket=bucket, route=request.url.path)
        raise RateLimited(decision.retry_after)


def user_rate_limit(bucket: str, allowance: Callable[[Settings], int]) -> Callable[..., None]:
    """Limit an authenticated route, keyed by the principal.

    **Takes `TenantDep` so the ordering is guaranteed rather than assumed.** A
    dependency that read the user off `request.state` would depend on FastAPI
    resolving authentication first; declaring the tenant as a parameter makes the
    dependency graph enforce it, and an unauthenticated caller is refused by
    `current_tenant` before any bucket is touched.

    `allowance` is a function of `Settings`, not a number, so the limit is read
    from the app actually serving the request — a value captured at import would
    ignore a test's configuration and silently ignore an operator's.
    """

    def dependency(request: Request, scope: TenantDep) -> None:
        _enforce(
            request,
            bucket=bucket,
            allowance=allowance(request.app.state.settings),
            key=f"user:{scope.user_id}",
        )

    # Read by the completeness test to see which routes are limited. An
    # attribute rather than a name comparison, because a test that matched
    # on "dependency" would silently start matching other closures.
    dependency.rate_limit_bucket = bucket  # type: ignore[attr-defined]
    return dependency


def peer_rate_limit(bucket: str, allowance: Callable[[Settings], int]) -> Callable[..., None]:
    """Limit an unauthenticated route, keyed by the socket peer.

    This is the half that matters most: `/v1/auth/login` and `/register` have no
    principal to key by, and they are what credential stuffing targets.
    """

    def dependency(request: Request) -> None:
        _enforce(
            request,
            bucket=bucket,
            allowance=allowance(request.app.state.settings),
            key=_peer_key(request),
        )

    # Read by the completeness test to see which routes are limited. An
    # attribute rather than a name comparison, because a test that matched
    # on "dependency" would silently start matching other closures.
    dependency.rate_limit_bucket = bucket  # type: ignore[attr-defined]
    return dependency


#: The buckets, named once so a route and its completeness test say the same
#: string. Auth is peer-keyed; the rest are user-keyed.
AUTH_BUCKET = "auth"
QA_BUCKET = "qa"
UPLOAD_BUCKET = "upload"
DISCOVERY_BUCKET = "discovery"
ADD_BUCKET = "add"

AuthRateLimit = Depends(peer_rate_limit(AUTH_BUCKET, lambda s: s.rate_limit_auth_per_minute))
QaRateLimit = Depends(user_rate_limit(QA_BUCKET, lambda s: s.rate_limit_qa_per_minute))
UploadRateLimit = Depends(user_rate_limit(UPLOAD_BUCKET, lambda s: s.rate_limit_upload_per_minute))
DiscoveryRateLimit = Depends(
    user_rate_limit(DISCOVERY_BUCKET, lambda s: s.rate_limit_discovery_per_minute)
)


def _add_rate_limit(request: Request, scope: TenantDep, body: AddDiscoveredPapers) -> None:
    """Limit adding discovered papers, counting each PAPER rather than the request (D133).

    Each paper is a lookup share, a paced fetch, a parse and an embedding pass, so
    ten papers in one request must cost what ten requests would. The body is the
    route's own (same name, same model), which FastAPI parses once; a malformed
    body is a 422 before this runs and spends nothing.
    """
    _enforce(
        request,
        bucket=ADD_BUCKET,
        allowance=request.app.state.settings.rate_limit_add_per_minute,
        key=f"user:{scope.user_id}",
        cost=len(body.papers),
    )


_add_rate_limit.rate_limit_bucket = ADD_BUCKET  # type: ignore[attr-defined]
AddRateLimit = Depends(_add_rate_limit)

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


def streaming_workspaces(session: StreamSessionDep, scope: TenantDep) -> SqlWorkspaceRepository:
    """`workspaces`, on the streamed answer's one session (D166)."""
    return SqlWorkspaceRepository(session, scope)


def streaming_answer_use_case(
    session: StreamSessionDep,
    scope: TenantDep,
    embedder: EmbedderDep,
    index: IndexDep,
    providers: ProvidersDep,
) -> AnswerQuestionUseCase:
    """`answer_use_case`, every repository on the one session that outlives the stream.

    Built here rather than from `WorkspacesDep` and its siblings, which are on
    `SessionDep`: that session closes when the handler returns, before the turns
    are written, and a request would otherwise hold two sessions.
    """
    return AnswerQuestionUseCase(
        workspaces=SqlWorkspaceRepository(session, scope),
        papers=SqlPaperRepository(session, scope),
        chunks=SqlChunkRepository(session, scope),
        conversations=SqlConversationRepository(session, scope),
        embedder=embedder,
        index=index,
        providers=providers,
    )


StreamingWorkspacesDep = Annotated[SqlWorkspaceRepository, Depends(streaming_workspaces)]
StreamingAnswerDep = Annotated[AnswerQuestionUseCase, Depends(streaming_answer_use_case)]

AddPaperDep = Annotated[AddPaperUseCase, Depends(add_paper_use_case)]
RemovePaperDep = Annotated[RemovePaperUseCase, Depends(remove_paper_use_case)]


def get_discovery_sources(request: Request) -> list[ConfiguredSource]:
    """The process's sources: built once, because each breaker and the arXiv pacer
    are state that must outlive a request (D104, D123)."""
    sources: list[ConfiguredSource] = request.app.state.discovery_sources
    return sources


def discover_use_case(request: Request) -> DiscoverPapersUseCase:
    settings: Settings = request.app.state.settings
    cache = getattr(request.app.state, "discovery_cache", None)
    embedder = getattr(request.app.state, "embedder", None)
    return DiscoverPapersUseCase(
        get_discovery_sources(request),
        # Absent means uncached, which is correct, not broken: a cache can only
        # ever turn a source call into a miss (D127).
        cache=cache,
        # Dense with the app's encoder, its work vectors in the same cache (D153);
        # lexical where no encoder is wired. A cache that is not a vector cache
        # (a test double) leaves the vectors uncached, never the search broken.
        relevance=(
            DenseRelevance(embedder, cache if isinstance(cache, VectorCache) else None)
            if embedder is not None
            else None
        ),
        # The add endpoint's allowlist, so the record each work offers is one add
        # will accept (D147).
        allowed_hosts=host_allowlist(settings.fetch_allowed_hosts),
        deadline_seconds=settings.discovery_deadline_seconds,
    )


DiscoverDep = Annotated[DiscoverPapersUseCase, Depends(discover_use_case)]


def add_discovered_use_case(
    request: Request,
    session: SessionDep,
    scope: TenantDep,
    workspace_repo: WorkspacesDep,
    paper_repo: PapersDep,
) -> AddDiscoveredPapersUseCase:
    settings: Settings = request.app.state.settings
    return AddDiscoveredPapersUseCase(
        workspaces=workspace_repo,
        papers=paper_repo,
        chunks=SqlChunkRepository(session, scope),
        # Same session as the repositories: papers, memberships and jobs are one
        # transaction (D47), so a batch that fails part-way leaves nothing (D133).
        queue=PostgresQueue(session, scope),
        # The process's sources, so a lookup passes through the same breakers and
        # the same arXiv pacer as a search (D123, D133).
        sources={c.source.name: c.source for c in get_discovery_sources(request)},
        allowed_hosts=host_allowlist(settings.fetch_allowed_hosts),
        owner_id=scope.user_id,
    )


AddDiscoveredDep = Annotated[AddDiscoveredPapersUseCase, Depends(add_discovered_use_case)]
