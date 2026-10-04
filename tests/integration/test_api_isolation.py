"""Layer 1 — the API's own refusal, with two real users (NFR-T5, FR-A2, D91).

`test_tenant_isolation.py` is the other half of NFR-T5, and its docstring says
why it never goes through the API: it asserts that layer 2 holds *even when a
route check is forgotten*. This file asserts the complementary thing — that the
route check is not forgotten today — and it needs what phase 1 made possible, two
users who can actually log in.

**Three properties, and only the first is about status codes.**

1. Every tenant-scoped endpoint answers a non-owner **404**, with a positive
   control beside it so a server that 404s unconditionally cannot pass.
2. The **body** carries nothing either. With the caller's own ids substituted
   out, the answer for "someone else's real id" is byte-identical to the answer
   for an id belonging to nobody. This is the property that found D93, where the
   status code was already 404 and the detail named the owning workspace.
3. A refused request **never reaches retrieval**. A 404 returned after an ANN
   query has already touched another tenant's paper ids is still a leak, and no
   status code can tell the two apart — so the vector index is a spy and the
   refused paths assert it was never called.

**Postgres only.** The app is built through `create_app`'s lifespan seam with a
deterministic embedder, the spy index and a stub provider, so this suite needs no
Qdrant, no GPU and no model, and therefore runs in the CI job where a mandatory
suite belongs. Layer 3's own leakage test drives a real ANN index against real
Qdrant, and D90 makes that one run in CI too.

Each test gets a freshly seeded world, which is why the endpoint table is
parametrised rather than looped: several of these calls are destructive, and a
single shared world would make the later rows depend on the earlier ones.
"""

from __future__ import annotations

import contextlib
import subprocess
import sys
import uuid
from collections.abc import AsyncIterator, Iterator, Sequence
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from typing import Any

import httpx
import pytest
import pytest_asyncio
from asgi_lifespan import LifespanManager
from fastapi import FastAPI
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from testcontainers.community.postgres import PostgresContainer

from scholiast.api.app import create_app
from scholiast.application.ports.llm import GenerationChunk
from scholiast.application.ports.ml import TextRole
from scholiast.application.ports.tenancy import TenantContext
from scholiast.application.ports.vector import EmptyRetrievalScope, VectorRecord
from scholiast.config.settings import Settings
from scholiast.domain.chunking import CHUNKER_VERSION
from scholiast.domain.conversation import Turn, TurnRole
from scholiast.domain.paper import (
    Chunk,
    ChunkRole,
    Paper,
    PaperVisibility,
    Provenance,
    SectionKind,
)
from scholiast.domain.retrieval import RetrievalArm, ScoredChunk
from scholiast.domain.workspace import PaperState, Workspace, WorkspaceState
from scholiast.infrastructure.db.engine import build_engine, build_session_factory
from scholiast.infrastructure.db.repositories import (
    SqlChunkRepository,
    SqlConversationRepository,
    SqlPaperRepository,
    SqlWorkspaceRepository,
)
from scholiast.infrastructure.queue.postgres import PostgresQueue
from scholiast.infrastructure.security.passwords import Argon2idHasher
from scholiast.infrastructure.security.tokens import JwtTokenService
from scholiast.infrastructure.storage.local import LocalObjectStore

pytestmark = [pytest.mark.integration]

ROOT = Path(__file__).resolve().parents[2]
DIMENSIONS = 32
PASSWORD = "a-perfectly-adequate-password"

#: A body that clears `validate_upload` (size plus the `%PDF-` signature) so the
#: upload endpoint reaches its workspace check. A body that failed validation
#: would be refused *before* authorisation and would test the wrong thing.
MINIMAL_PDF = b"%PDF-1.4\n% just valid enough to reach the workspace check\n%%EOF\n"


@dataclass(frozen=True)
class Endpoint:
    """One tenant-scoped route, as a template over a tenant's ids."""

    label: str
    method: str
    path: str
    json: dict[str, object] | None = None
    #: A multipart upload rather than a JSON body.
    upload: bool = False
    #: What the *owner* gets. Asserted exactly, because "not 404" would pass on a
    #: 500 and the status code is the part a client branches on.
    owner_status: int = 200

    def resolve(self, ids: Ids) -> str:
        return self.path.format(
            workspace=ids.workspace_id,
            paper=ids.paper_id,
            conversation=ids.conversation_id,
            job=ids.job_id,
        )

    async def send(self, client: httpx.AsyncClient, ids: Ids) -> httpx.Response:
        path = self.resolve(ids)
        if self.upload:
            return await client.post(
                path, files={"file": ("paper.pdf", MINIMAL_PDF, "application/pdf")}
            )
        return await client.request(self.method, path, json=self.json)


#: Every tenant-scoped endpoint the API exposes. `auth` and `health` are absent
#: because they have no tenant, and `POST`/`GET /v1/workspaces` are absent
#: because there is no owner to be a non-owner of — a create and a listing.
#: `test_every_tenant_scoped_route_is_covered` fails if a route is added and this
#: table is not, which is what keeps NFR-T5's "every endpoint" true over time.
ENDPOINTS: tuple[Endpoint, ...] = (
    Endpoint("read a workspace", "GET", "/v1/workspaces/{workspace}"),
    Endpoint(
        "rename a workspace",
        "PATCH",
        "/v1/workspaces/{workspace}",
        json={"name": "renamed by a stranger"},
    ),
    # 202, not 204: §5.6.4 makes marking DELETING the only synchronous step.
    Endpoint("delete a workspace", "DELETE", "/v1/workspaces/{workspace}", owner_status=202),
    Endpoint("list papers", "GET", "/v1/workspaces/{workspace}/papers"),
    Endpoint(
        "upload a paper",
        "POST",
        "/v1/workspaces/{workspace}/papers/upload",
        upload=True,
        owner_status=202,
    ),
    Endpoint("read paper content", "GET", "/v1/workspaces/{workspace}/papers/{paper}/content"),
    Endpoint("remove a paper", "DELETE", "/v1/workspaces/{workspace}/papers/{paper}"),
    Endpoint(
        "open a conversation",
        "POST",
        "/v1/workspaces/{workspace}/conversations",
        json={"title": None},
        owner_status=201,
    ),
    Endpoint(
        "read a transcript",
        "GET",
        "/v1/workspaces/{workspace}/conversations/{conversation}/messages",
    ),
    Endpoint(
        "ask a question",
        "POST",
        "/v1/workspaces/{workspace}/conversations/{conversation}/messages",
        json={"question": "What does their confidential method do?"},
    ),
    Endpoint("read a job", "GET", "/v1/jobs/{job}"),
)

LABELS = [endpoint.label for endpoint in ENDPOINTS]


@dataclass(frozen=True)
class Ids:
    """The identifiers one tenant owns — or, for a ghost, that nobody owns."""

    workspace_id: str
    paper_id: str
    conversation_id: str
    job_id: str

    @classmethod
    def nobodys(cls) -> Ids:
        return cls(*(str(uuid.uuid4()) for _ in range(4)))

    def all(self) -> tuple[str, ...]:
        return (self.workspace_id, self.paper_id, self.conversation_id, self.job_id)


@dataclass
class Tenant:
    client: httpx.AsyncClient
    user_id: str
    ids: Ids
    #: A second workspace, so a destructive positive control has somewhere to
    #: land that no other assertion reads.
    spare_workspace_id: str


@dataclass
class World:
    alice: Tenant
    bob: Tenant
    index: SpyVectorIndex
    app: FastAPI


@pytest.fixture(scope="module")
def postgres() -> Iterator[str]:
    with PostgresContainer("postgres:17-alpine", driver="asyncpg") as container:
        url = container.get_connection_url()
        result = subprocess.run(  # noqa: S603 - fixed argv, no shell
            [sys.executable, "-m", "alembic", "-x", f"db_url={url}", "upgrade", "head"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode != 0:
            pytest.fail(f"alembic upgrade failed:\n{result.stdout}\n{result.stderr}")
        yield url


class DeterministicEmbedder:
    version = "test-embedder/1.0.0-fp32"
    dimensions = DIMENSIONS

    async def embed(
        self, texts: Sequence[str], *, role: TextRole, batch_size: int | None = None
    ) -> list[list[float]]:
        return [
            [((abs(hash(t)) % 9973 + i * 37) % 100) / 100.0 for i in range(DIMENSIONS)]
            for t in texts
        ]


class SpyVectorIndex:
    """Records every scope it is asked to search, and answers from a canned map.

    The recording is the point. A 404 produced *after* an ANN query has run
    against another tenant's paper ids is a leak that a status-code assertion
    cannot see, so the refused paths assert `searches == []` rather than trusting
    the order of checks inside the use case.
    """

    def __init__(self) -> None:
        self.searches: list[tuple[str, ...]] = []
        self.hits: dict[str, str] = {}  # paper_id -> chunk_id
        #: Set to make the index answer with a chunk OUTSIDE the scope it was
        #: given — a vector store that disagrees with Postgres, which §5.6.4 says
        #: must not become a leak.
        self.leak: tuple[str, str] | None = None  # (chunk_id, paper_id)

    async def ensure_ready(self) -> None:
        return None

    async def close(self) -> None:
        return None

    async def upsert(self, records: Sequence[VectorRecord]) -> int:
        return len(records)

    async def search(
        self, vector: Sequence[float], *, paper_ids: Sequence[str], limit: int
    ) -> list[ScoredChunk]:
        if not paper_ids:
            # The port's contract: an empty scope is an error, never a scan.
            raise EmptyRetrievalScope
        self.searches.append(tuple(paper_ids))
        if self.leak is not None:
            chunk_id, paper_id = self.leak
            return [
                ScoredChunk(
                    chunk_id=chunk_id, paper_id=paper_id, score=1.0, arm=RetrievalArm.DENSE, rank=1
                )
            ]
        return [
            ScoredChunk(
                chunk_id=self.hits[paper_id],
                paper_id=paper_id,
                score=0.9,
                arm=RetrievalArm.DENSE,
                rank=rank + 1,
            )
            for rank, paper_id in enumerate(paper_ids)
            if paper_id in self.hits
        ][:limit]

    async def delete_by_papers(self, paper_ids: Sequence[str]) -> int:
        return 0

    async def count(self, *, paper_ids: Sequence[str] | None = None) -> int:
        return len(self.hits)


class StubProvider:
    name = "stub"
    model = "stub-model"

    async def stream(self, *, system: str, user: str) -> AsyncIterator[GenerationChunk]:
        yield GenerationChunk(text="An answer grounded in [1]. ")
        yield GenerationChunk(done=True)


def _lifespan(index: SpyVectorIndex) -> Any:
    """`api_lifespan` minus everything tenancy does not need.

    The embedder, the vector index and the provider chain are the only
    substitutions, and each replaces a dependency that would otherwise require a
    419 MB encoder, Qdrant, or a running Ollama. The engine, the hasher and the
    token service are the real ones, because this suite logs in for real.
    """

    @contextlib.asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        settings: Settings = app.state.settings
        engine = build_engine(settings)
        app.state.engine = engine
        app.state.sessions = build_session_factory(engine)
        app.state.store = LocalObjectStore(settings.object_store_path)
        app.state.hasher = Argon2idHasher()
        app.state.tokens = JwtTokenService(
            secret=settings.jwt_secret,
            issuer=settings.jwt_issuer,
            access_ttl=timedelta(minutes=settings.access_token_minutes),
            refresh_ttl=timedelta(days=settings.refresh_token_days),
        )
        app.state.embedder = DeterministicEmbedder()
        app.state.index = index
        app.state.providers = [StubProvider()]
        try:
            yield
        finally:
            await engine.dispose()

    return lifespan


async def _seed(
    session: AsyncSession, *, label: str, owner_id: str, index: SpyVectorIndex
) -> tuple[Ids, str]:
    """One registered user's world, written through layer 2 under their own scope.

    Seeded through the repositories rather than the API because the ingestion path
    wants GROBID and a worker to reach INDEXED; what this suite asserts is the
    refusal, and every assertion goes over HTTP.
    """
    scope = TenantContext(user_id=owner_id)
    workspaces = SqlWorkspaceRepository(session, scope)
    made = [
        await workspaces.create(
            Workspace(
                id=str(uuid.uuid4()),
                owner_id=owner_id,
                name=name,
                state=WorkspaceState.ACTIVE,
            )
        )
        for name in (f"{label}'s workspace", f"{label}'s spare")
    ]
    workspace, spare = made

    papers = SqlPaperRepository(session, scope)
    paper = await papers.add(
        Paper(
            id=str(uuid.uuid4()),
            # Byte-identical across both tenants: §4.4.2's carve-out says these
            # stay two private rows rather than collapsing into one shared one.
            content_hash="identical-manuscript-bytes",
            visibility=PaperVisibility.PRIVATE,
            owner_id=owner_id,
            title=f"{label}'s Unpublished Manuscript",
            parser_version="grobid-crf/0.8.2.1",
        )
    )
    await papers.add_to_workspace(workspace.id, paper.id, PaperState.INDEXED)

    section_id = uuid.uuid4()
    await session.execute(
        text(
            "INSERT INTO sections (id, paper_id, kind, raw_title, path, order_index, "
            "page_start, page_end, text) VALUES (:id, :paper, 'method', 'Method', "
            "'Method', 0, 1, 3, :body)"
        ),
        {"id": section_id, "paper": uuid.UUID(paper.id), "body": f"{label} confidential method"},
    )
    parent = f"{paper.id}:p0"
    await session.execute(
        text(
            "INSERT INTO chunks (id, paper_id, section_id, role, ordinal, text, token_count, "
            "page, char_start, char_end, chunker_version) VALUES (:id, :paper, :section, "
            "'parent', 0, 'window', 40, 1, 0, 40, :ver)"
        ),
        {"id": parent, "paper": uuid.UUID(paper.id), "section": section_id, "ver": CHUNKER_VERSION},
    )
    chunk_id = f"{parent}:c0"
    await SqlChunkRepository(session, scope).add_many(
        [
            Chunk(
                id=chunk_id,
                text=f"{label} confidential method",
                role=ChunkRole.CHILD,
                ordinal=1,
                provenance=Provenance(
                    paper_id=paper.id,
                    section_path="Method",
                    section_kind=SectionKind.METHOD,
                    page=1,
                    char_start=0,
                    char_end=20,
                ),
                chunker_version=CHUNKER_VERSION,
                token_count=12,
                parent_id=parent,
            )
        ],
        section_ids={"Method": str(section_id)},
    )
    index.hits[paper.id] = chunk_id

    conversations = SqlConversationRepository(session, scope)
    conversation_id = await conversations.create(workspace.id, title=f"{label}'s thread")
    assert conversation_id is not None
    await conversations.append(
        conversation_id, Turn(role=TurnRole.USER, content=f"{label}'s private question")
    )

    job_id = await PostgresQueue(session, scope).enqueue(
        workspace_id=workspace.id, paper_id=paper.id
    )
    await session.commit()
    return (
        Ids(
            workspace_id=workspace.id,
            paper_id=paper.id,
            conversation_id=conversation_id,
            job_id=job_id,
        ),
        spare.id,
    )


@pytest_asyncio.fixture
async def world(postgres: str, tmp_path: Path) -> AsyncIterator[World]:
    index = SpyVectorIndex()
    app = create_app(
        Settings(
            database_url=postgres,  # type: ignore[arg-type]
            object_store_path=tmp_path / "objects",
            embedding_dimensions=DIMENSIONS,
            worker_in_process=False,
        ),
        lifespan=_lifespan(index),
    )
    engine = create_async_engine(postgres)
    factory = async_sessionmaker(engine, expire_on_commit=False)

    async with LifespanManager(app) as manager:
        transport = httpx.ASGITransport(app=manager.app)
        clients: list[httpx.AsyncClient] = []
        tenants: dict[str, Tenant] = {}

        for label in ("alice", "bob"):
            client = httpx.AsyncClient(transport=transport, base_url="http://test")
            clients.append(client)
            # Registered, not seeded: the principal has to come from a real token
            # for this to be a layer-1 test at all.
            registered = await client.post(
                "/v1/auth/register",
                json={
                    "email": f"{label}-{uuid.uuid4().hex[:10]}@example.com",
                    "password": PASSWORD,
                },
            )
            assert registered.status_code == 201, registered.text
            body = registered.json()
            client.headers["Authorization"] = f"Bearer {body['access_token']}"

            async with factory() as session:
                ids, spare = await _seed(
                    session, label=label, owner_id=body["user_id"], index=index
                )
            tenants[label] = Tenant(
                client=client, user_id=body["user_id"], ids=ids, spare_workspace_id=spare
            )

        yield World(alice=tenants["alice"], bob=tenants["bob"], index=index, app=app)

        for client in clients:
            await client.aclose()

    async with engine.begin() as connection:
        for table in (
            "messages",
            "conversations",
            "ingestion_jobs",
            "chunks",
            "sections",
            "workspace_papers",
            "papers",
            "workspaces",
            "refresh_tokens",
            "users",
        ):
            await connection.execute(text(f"DELETE FROM {table}"))  # noqa: S608 - fixed list
    await engine.dispose()


def _redacted(response: httpx.Response, ids: Sequence[str]) -> tuple[int, str]:
    """The response with the caller's own identifiers substituted out.

    What is left is everything the server *chose* to say. Comparing that between
    "someone else's real id" and "an id belonging to nobody" is what makes the
    404 claim a claim about the body and not just the status line — the
    distinction D93 turned on.
    """
    body = response.text
    for position, value in enumerate(ids):
        body = body.replace(value, f"<id-{position}>")
    return (response.status_code, body)


@pytest.mark.parametrize("endpoint", ENDPOINTS, ids=LABELS)
class TestEveryEndpointRefusesANonOwner:
    """FR-A2: 404, never 403 — and never a 2xx."""

    async def test_a_non_owner_is_refused(self, world: World, endpoint: Endpoint) -> None:
        response = await endpoint.send(world.alice.client, world.bob.ids)
        assert response.status_code == 404, response.text

    async def test_the_owner_is_not(self, world: World, endpoint: Endpoint) -> None:
        # The positive control. Without it every assertion above would pass
        # against a server that answers 404 unconditionally.
        target = world.bob.ids
        if endpoint.method == "DELETE" and endpoint.path == "/v1/workspaces/{workspace}":
            # Deleting the workspace under test would invalidate nothing here —
            # each test gets a fresh world — but the spare says so explicitly.
            target = Ids(
                workspace_id=world.bob.spare_workspace_id,
                paper_id=target.paper_id,
                conversation_id=target.conversation_id,
                job_id=target.job_id,
            )
        response = await endpoint.send(world.bob.client, target)
        assert response.status_code == endpoint.owner_status, response.text

    async def test_the_refusal_is_identical_to_one_for_an_id_nobody_owns(
        self, world: World, endpoint: Endpoint
    ) -> None:
        ghost = Ids.nobodys()
        theirs = _redacted(
            await endpoint.send(world.alice.client, world.bob.ids), world.bob.ids.all()
        )
        nobodys = _redacted(await endpoint.send(world.alice.client, ghost), ghost.all())
        assert theirs == nobodys

    async def test_no_refusal_names_something_the_caller_never_sent(
        self, world: World, endpoint: Endpoint
    ) -> None:
        """The specific leak D93 fixed, asserted for every endpoint.

        `GET /v1/jobs/{id}` answered with the *owning workspace's* uuid, which the
        caller never supplied and could not otherwise learn. The path's own
        segments are excluded because echoing back what was asked about is a
        diagnostic, not a disclosure.
        """
        supplied = set(endpoint.resolve(world.bob.ids).split("/"))
        secrets = (
            world.bob.ids.workspace_id,
            world.bob.spare_workspace_id,
            world.bob.user_id,
            "confidential",
        )
        response = await endpoint.send(world.alice.client, world.bob.ids)
        leaked = [s for s in secrets if s in response.text and s not in supplied]
        assert leaked == [], f"{endpoint.label} leaked {leaked}"


class TestListingsAndCreates:
    """The two routes with no owner to be a non-owner of."""

    async def test_a_listing_shows_only_your_own_workspaces(self, world: World) -> None:
        listed = await world.alice.client.get("/v1/workspaces")
        assert listed.status_code == 200
        ids = {row["id"] for row in listed.json()}
        assert world.alice.ids.workspace_id in ids
        assert ids.isdisjoint({world.bob.ids.workspace_id, world.bob.spare_workspace_id})

    async def test_a_created_workspace_belongs_to_its_creator(self, world: World) -> None:
        created = await world.alice.client.post("/v1/workspaces", json={"name": "mine"})
        assert created.status_code == 201
        # And it is invisible to the other tenant immediately.
        assert (
            await world.bob.client.get(f"/v1/workspaces/{created.json()['id']}")
        ).status_code == 404


class TestIdentifiersMustAgreeWithTheirPath:
    """A pair of ids that both belong to you, but not to each other.

    Not a tenant leak — both ids are the caller's — which is why a mutation run
    found this untested: dropping the check in `list_messages` left every
    cross-tenant assertion passing, because `ConversationRepository.get` is
    already scoped and answers None for someone else's thread. What that clause
    actually defends is the pairing, and a route that ignores its own path
    parameter is how a mismatched reference becomes a silent read of the wrong
    workspace's history.

    **The answering route does NOT make the same check**, and that is recorded
    rather than fixed here: `POST .../conversations/{id}/messages` authorises the
    workspace, retrieves within it, and appends the turn to whatever conversation
    the path names, without comparing the two. Probing it returns 409 rather than
    200 only because FR-W5's queryability guard fires first on a workspace with
    no indexed papers. Both ids are the caller's own, so no isolation claim is
    affected — which is why it is a finding for a later phase and not a tenancy
    fix smuggled into this one.
    """

    async def test_a_conversation_cannot_be_read_through_another_workspace_path(
        self, world: World
    ) -> None:
        own_but_wrong_pair = (
            f"/v1/workspaces/{world.alice.spare_workspace_id}/conversations/"
            f"{world.alice.ids.conversation_id}/messages"
        )
        assert (await world.alice.client.get(own_but_wrong_pair)).status_code == 404

        # The positive control: the same conversation under its own workspace.
        correct = (
            f"/v1/workspaces/{world.alice.ids.workspace_id}/conversations/"
            f"{world.alice.ids.conversation_id}/messages"
        )
        assert (await world.alice.client.get(correct)).status_code == 200


class TestRefusedRequestsNeverReachRetrieval:
    """A 404 issued after an ANN query has already run is still a leak."""

    async def test_asking_in_another_workspace_never_searches(self, world: World) -> None:
        world.index.searches.clear()
        response = await world.alice.client.post(
            f"/v1/workspaces/{world.bob.ids.workspace_id}/conversations/"
            f"{world.bob.ids.conversation_id}/messages",
            json={"question": "What does their confidential method do?"},
        )
        assert response.status_code == 404
        assert world.index.searches == []

    async def test_the_owner_reaches_retrieval_scoped_to_their_own_papers(
        self, world: World
    ) -> None:
        # The positive control, and the §5.8 layer-3 mechanism assertion: the
        # scope handed to the index is exactly the caller's own paper ids, which
        # is what D36 replaced the workspace filter with.
        world.index.searches.clear()
        response = await world.bob.client.post(
            f"/v1/workspaces/{world.bob.ids.workspace_id}/conversations/"
            f"{world.bob.ids.conversation_id}/messages",
            json={"question": "What does the method do?"},
        )
        assert response.status_code == 200
        assert world.index.searches == [(world.bob.ids.paper_id,)]
        assert world.alice.ids.paper_id not in world.index.searches[0]

    async def test_another_tenants_text_never_appears_in_an_answer(self, world: World) -> None:
        # The end of the chain: alice's section text is in the database, the spy
        # would happily return it if asked, and the answer must still not contain
        # it. This is FR-A4 as a client sees it.
        streamed = await world.bob.client.post(
            f"/v1/workspaces/{world.bob.ids.workspace_id}/conversations/"
            f"{world.bob.ids.conversation_id}/messages",
            json={"question": "Summarise the method."},
        )
        assert streamed.status_code == 200
        assert "alice confidential method" not in streamed.text


class TestTheStoresDisagreeing:
    """§5.6.4: Qdrant and Postgres can drift, and drift must not become a leak.

    `AnswerQuestionUseCase._resolve` fetches every retrieved chunk back through
    the scoped repository precisely so a vector the index should not have
    returned is dropped rather than served. That guard had no test: the spy makes
    the failure reproducible by answering with another tenant's chunk id however
    narrow the scope it was handed.
    """

    async def test_a_foreign_chunk_returned_by_the_index_is_dropped(self, world: World) -> None:
        world.index.leak = (f"{world.alice.ids.paper_id}:p0:c0", world.alice.ids.paper_id)

        response = await world.bob.client.post(
            f"/v1/workspaces/{world.bob.ids.workspace_id}/conversations/"
            f"{world.bob.ids.conversation_id}/messages",
            json={"question": "Summarise the method."},
        )

        assert response.status_code == 200
        # The index WAS asked, and only for bob's papers — so the leak came from
        # the store, not from a widened scope.
        assert world.index.searches[-1] == (world.bob.ids.paper_id,)
        # And alice's text never reaches the wire.
        assert "alice confidential method" not in response.text
        assert world.alice.ids.paper_id not in response.text


class TestTheTableIsComplete:
    async def test_every_tenant_scoped_route_is_covered(self, world: World) -> None:
        """A new endpoint must join `ENDPOINTS` or this fails.

        NFR-T5's claim is "every endpoint", and a hand-maintained list rots the
        moment someone adds a route. Comparing it with the app's own contract is
        what keeps the claim true rather than true-when-written.

        **The source is the OpenAPI document, not `app.routes`.** The first
        version of this test walked `app.routes`, where this FastAPI version
        keeps each `include_router` as an `_IncludedRouter` with no `.path` and
        no `.methods` — so the routed set came out EMPTY and the assertion
        passed against anything, including a table with an endpoint deleted. A
        mutation run is what exposed it. The non-emptiness assertion below is the
        structural fix: a completeness test that cannot fail is worse than no
        completeness test, because it reads as evidence.
        """
        document = world.app.openapi()
        routed = {
            (method.upper(), path)
            for path, operations in document["paths"].items()
            for method in operations
            if path.startswith("/v1") and not path.startswith(("/v1/auth", "/v1/health"))
        }
        assert len(routed) >= len(ENDPOINTS), f"the routing table came out empty or short: {routed}"

        covered = {
            (
                endpoint.method,
                endpoint.path.format(
                    workspace="{workspace_id}",
                    paper="{paper_id}",
                    conversation="{conversation_id}",
                    job="{job_id}",
                ),
            )
            for endpoint in ENDPOINTS
        }
        # The two routes with no owner to be a non-owner of — a listing and a
        # create — are covered by `TestListingsAndCreates` instead.
        covered |= {("GET", "/v1/workspaces"), ("POST", "/v1/workspaces")}
        # Discovery reads no tenant's data: it asks external sources and writes
        # nothing (§5.4.1), so there is no owner to be a non-owner of. It still
        # requires a user, for the rate-limit key (D113, D122). Phase 4's shared
        # cache is where one tenant's search could become visible to another, and
        # that exposure is tested there, not assumed away here.
        covered |= {("POST", "/v1/discovery/search")}
        assert routed - covered == set(), f"uncovered routes: {routed - covered}"
