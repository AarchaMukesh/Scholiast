"""The HTTP surface (§5.5, FR-W1, FR-W2, NFR-SEC2).

Separate from the acceptance suite: that file asserts the *milestone's* criteria,
this one asserts the API behaves like an API — status codes, problem documents,
idempotence, and the tenancy boundary as seen from outside.

Most of this needs only Postgres and Qdrant. GROBID and Ollama are required only
where a real paper or a real answer is, and are marked accordingly.
"""

from __future__ import annotations

import os
import subprocess
import sys
import uuid
from collections.abc import AsyncIterator, Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path

import httpx
import pytest
import pytest_asyncio
from asgi_lifespan import LifespanManager
from fastapi import FastAPI
from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine
from testcontainers.community.postgres import PostgresContainer

from scholiast.api.app import create_app
from scholiast.application.ports.ml import TextRole
from scholiast.application.ports.tenancy import TenantContext
from scholiast.config.settings import Settings
from scholiast.infrastructure.db.repositories import SqlWorkspaceRepository

pytestmark = [pytest.mark.integration]

ROOT = Path(__file__).resolve().parents[2]
FIXTURES = ROOT / "spike" / "fixtures"
GROBID_URL = "http://localhost:8070"
QDRANT_URL = "http://localhost:6333"
DIMENSIONS = 32


#: Migration 0005 made `users.password_hash` NOT NULL (D85). These rows exist
#: to own data, never to log in, so they get a string that is not a valid
#: Argon2 encoding: `Argon2idHasher.verify` returns False for it against any
#: password, so the account is unreachable by construction rather than by
#: nobody knowing the password.
UNUSABLE_HASH = "not-a-real-argon2-hash"


def _alive(url: str) -> bool:
    try:
        with httpx.Client(timeout=5.0) as client:
            return client.get(url).status_code == 200
    except httpx.HTTPError:
        return False


needs_qdrant = pytest.mark.skipif(
    not _alive(f"{QDRANT_URL}/readyz"), reason="Qdrant is not running"
)
needs_grobid = pytest.mark.skipif(
    not _alive(f"{GROBID_URL}/api/isalive"), reason="GROBID is not running"
)
needs_fixtures = pytest.mark.skipif(not FIXTURES.exists(), reason="fixture PDFs are gitignored")


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


@dataclass
class Harness:
    client: httpx.AsyncClient
    app: FastAPI
    postgres: str


#: Migration 0005 gives the seeded M1 account a development password read from
#: the environment (D85). It is set explicitly here rather than left to the
#: documented default, so a developer who exports the variable does not silently
#: break this suite — and so these tests authenticate as the SAME principal M1's
#: `current_tenant` used to return unconditionally, which is what keeps the
#: ownership assertions below meaning what they meant.
SEED_EMAIL = "demo@scholiast.local"
SEED_PASSWORD = "integration-suite-password"


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
            env={**os.environ, "SEED_USER_PASSWORD": SEED_PASSWORD},
        )
        if result.returncode != 0:
            pytest.fail(f"alembic upgrade failed:\n{result.stdout}\n{result.stderr}")
        yield url


async def authenticate(client: httpx.AsyncClient) -> str:
    """Log in as the seeded account and attach the bearer token to the client.

    M3 phase 1 replaced the inside of `current_tenant`: every router now needs a
    verified token, and no router changed to gain that. The tests therefore do
    what any client does — they log in.
    """
    response = await client.post(
        "/v1/auth/login", json={"email": SEED_EMAIL, "password": SEED_PASSWORD}
    )
    assert response.status_code == 200, response.text
    token = response.json()["access_token"]
    client.headers["Authorization"] = f"Bearer {token}"
    return str(response.json()["user_id"])


@pytest_asyncio.fixture
async def api(postgres: str, tmp_path: Path) -> AsyncIterator[Harness]:
    collection = f"api_{uuid.uuid4().hex[:8]}"
    app = create_app(
        Settings(
            database_url=postgres,  # type: ignore[arg-type]
            qdrant_url=QDRANT_URL,
            qdrant_collection=collection,
            grobid_url=GROBID_URL,
            object_store_path=tmp_path / "objects",
            embedding_dimensions=DIMENSIONS,
            worker_in_process=False,
        )
    )
    async with LifespanManager(app) as manager:
        app.state.embedder = DeterministicEmbedder()
        transport = httpx.ASGITransport(app=manager.app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            await authenticate(client)
            yield Harness(client=client, app=app, postgres=postgres)

        engine = create_async_engine(postgres)
        async with engine.begin() as connection:
            for table in (
                "refresh_tokens",
                "messages",
                "conversations",
                "ingestion_jobs",
                "chunks",
                "sections",
                "workspace_papers",
                "papers",
                "workspaces",
            ):
                await connection.execute(text(f"DELETE FROM {table}"))  # noqa: S608 - fixed list
        await engine.dispose()
        await app.state.index._client.delete_collection(collection)


async def drain(app: FastAPI, limit: int = 6) -> int:
    from scholiast.infrastructure.parsing.grobid import GrobidParser
    from scholiast.workers.ingestion import IngestionWorker, WorkerConfig

    worker = IngestionWorker(
        sessions=app.state.sessions,
        store=app.state.store,
        parser=GrobidParser(GROBID_URL),
        embedder=app.state.embedder,
        index=app.state.index,
        config=WorkerConfig(poll_seconds=0.05),
    )
    total = 0
    for _ in range(limit):
        ran = await worker.run_once()
        if not ran:
            break
        total += ran
    return total


@needs_qdrant
class TestWorkspaceCrud:
    """FR-W1: create, rename, list, delete."""

    async def test_the_four_verbs_round_trip(self, api: Harness) -> None:
        created = await api.client.post("/v1/workspaces", json={"name": "Reading list"})
        assert created.status_code == 201
        workspace_id = created.json()["id"]

        listed = await api.client.get("/v1/workspaces")
        assert [w["id"] for w in listed.json()] == [workspace_id]

        renamed = await api.client.patch(f"/v1/workspaces/{workspace_id}", json={"name": "Renamed"})
        assert renamed.status_code == 200
        assert renamed.json()["name"] == "Renamed"

        # 202, not 204: §5.6.4 makes marking DELETING the only synchronous step,
        # and the rows are still there until M3's sweeper runs. Claiming 204
        # would say the deletion finished.
        deleted = await api.client.delete(f"/v1/workspaces/{workspace_id}")
        assert deleted.status_code == 202

    async def test_a_deleted_workspace_stops_serving_immediately(self, api: Harness) -> None:
        # Soft-deleted is invisible to every scoped read from the moment it is
        # marked — not when the sweeper eventually runs.
        created = await api.client.post("/v1/workspaces", json={"name": "Doomed"})
        workspace_id = created.json()["id"]
        await api.client.delete(f"/v1/workspaces/{workspace_id}")

        assert (await api.client.get(f"/v1/workspaces/{workspace_id}")).status_code == 404
        assert (await api.client.get("/v1/workspaces")).json() == []

    async def test_an_unknown_workspace_is_a_problem_document(self, api: Harness) -> None:
        response = await api.client.get(f"/v1/workspaces/{uuid.uuid4()}")
        assert response.status_code == 404
        assert response.headers["content-type"].startswith("application/problem+json")
        assert response.json()["type"] == "/problems/workspace-not-found"

    async def test_a_malformed_id_is_not_a_server_error(self, api: Harness) -> None:
        # Ids arrive from URLs. A probe for `../etc/passwd` must be a 404, not a
        # 500 that discloses a stack trace.
        response = await api.client.get("/v1/workspaces/not-a-uuid")
        assert response.status_code == 404


@needs_qdrant
class TestUploadValidationHappensAtTheEdge:
    """D55: refuse what can be refused without opening the file."""

    async def test_a_non_pdf_is_rejected_with_415(self, api: Harness) -> None:
        created = await api.client.post("/v1/workspaces", json={"name": "W"})
        workspace_id = created.json()["id"]

        response = await api.client.post(
            f"/v1/workspaces/{workspace_id}/papers/upload",
            files={"file": ("notes.pdf", b"PK\x03\x04 not a pdf", "application/pdf")},
        )
        assert response.status_code == 415
        body = response.json()
        assert body["failure_class"] == "unparseable"
        # NFR-U2: what the user can do, not what the parser saw.
        assert "password-protected or corrupted" in body["detail"]

    async def test_nothing_is_stored_for_a_rejected_upload(self, api: Harness) -> None:
        created = await api.client.post("/v1/workspaces", json={"name": "W"})
        workspace_id = created.json()["id"]
        await api.client.post(
            f"/v1/workspaces/{workspace_id}/papers/upload",
            files={"file": ("x.pdf", b"nope", "application/pdf")},
        )
        papers = await api.client.get(f"/v1/workspaces/{workspace_id}/papers")
        assert papers.json() == []

    async def test_uploading_into_another_tenants_workspace_is_404(self, api: Harness) -> None:
        # The tenancy boundary as seen from outside. The workspace exists; it is
        # simply not this caller's, and the response cannot say which (FR-A2).
        stranger = TenantContext(user_id=str(uuid.uuid4()))
        async with api.app.state.sessions() as session:
            await session.execute(
                text("INSERT INTO users (id, email, password_hash) VALUES (:id, :email, :hash)"),
                {
                    "id": stranger.user_id,
                    "email": f"{stranger.user_id}@x.test",
                    "hash": UNUSABLE_HASH,
                },
            )
            await session.commit()
        async with api.app.state.sessions() as session:
            from scholiast.domain.workspace import Workspace

            other = await SqlWorkspaceRepository(session, stranger).create(
                Workspace(id=str(uuid.uuid4()), owner_id=stranger.user_id, name="Theirs")
            )
            await session.commit()

        response = await api.client.post(
            f"/v1/workspaces/{other.id}/papers/upload",
            files={"file": ("x.pdf", b"%PDF-1.7 body", "application/pdf")},
        )
        assert response.status_code == 404


@needs_qdrant
@needs_grobid
@needs_fixtures
class TestPaperRemovalLeavesNoOrphans:
    """FR-W2, and its stated test: orphan detection after removal (D62)."""

    async def _ingest(self, api: Harness, name: str = "attention-is-all-you-need.pdf"):  # type: ignore[no-untyped-def]
        created = await api.client.post("/v1/workspaces", json={"name": "W"})
        workspace_id = created.json()["id"]
        upload = await api.client.post(
            f"/v1/workspaces/{workspace_id}/papers/upload",
            files={"file": (name, (FIXTURES / name).read_bytes(), "application/pdf")},
        )
        await drain(api.app)
        return workspace_id, upload.json()["paper_id"]

    async def test_removing_the_last_holder_purges_chunks_and_vectors(self, api: Harness) -> None:
        workspace_id, paper_id = await self._ingest(api)
        before = await api.app.state.index.count(paper_ids=[paper_id])
        assert before > 0

        response = await api.client.delete(f"/v1/workspaces/{workspace_id}/papers/{paper_id}")
        assert response.status_code == 200
        assert response.json() == {
            "paper_id": paper_id,
            "removed": True,
            "purged": True,
            "vectors_deleted": before,
        }

        # The orphan check, in every store FR-W2 names.
        assert await api.app.state.index.count(paper_ids=[paper_id]) == 0
        engine = create_async_engine(api.postgres)
        async with engine.connect() as connection:
            counts = {
                table: (
                    await connection.execute(
                        text(f"SELECT count(*) FROM {table} WHERE paper_id = :p"),  # noqa: S608
                        {"p": paper_id},
                    )
                ).scalar_one()
                for table in ("chunks", "sections", "workspace_papers")
            }
            papers = (
                await connection.execute(
                    text("SELECT count(*) FROM papers WHERE id = :p"), {"p": paper_id}
                )
            ).scalar_one()
        await engine.dispose()
        assert counts == {"chunks": 0, "sections": 0, "workspace_papers": 0}
        assert papers == 0

    async def test_the_blobs_go_too(self, api: Harness) -> None:
        from scholiast.application.ports.storage import source_key

        workspace_id, paper_id = await self._ingest(api)
        assert await api.app.state.store.exists(source_key(paper_id))

        await api.client.delete(f"/v1/workspaces/{workspace_id}/papers/{paper_id}")
        assert not await api.app.state.store.exists(source_key(paper_id))

    async def test_removal_is_idempotent(self, api: Harness) -> None:
        # NFR-R5: a retried delete must not fail for having already succeeded.
        workspace_id, paper_id = await self._ingest(api)
        first = await api.client.delete(f"/v1/workspaces/{workspace_id}/papers/{paper_id}")
        second = await api.client.delete(f"/v1/workspaces/{workspace_id}/papers/{paper_id}")

        assert first.json()["removed"] is True
        assert second.status_code == 200
        assert second.json()["removed"] is False
        assert second.json()["purged"] is False

    async def test_a_paper_another_workspace_still_holds_is_not_purged(self, api: Harness) -> None:
        # AD-8: the registry shares a paper so it is parsed once. Removing it
        # from one workspace must not break retrieval in the other — which is
        # why `has_any_membership` deliberately looks across all of them.
        pdf = (FIXTURES / "attention-is-all-you-need.pdf").read_bytes()
        first = (await api.client.post("/v1/workspaces", json={"name": "A"})).json()["id"]
        second = (await api.client.post("/v1/workspaces", json={"name": "B"})).json()["id"]

        upload = await api.client.post(
            f"/v1/workspaces/{first}/papers/upload",
            files={"file": ("a.pdf", pdf, "application/pdf")},
        )
        paper_id = upload.json()["paper_id"]
        await drain(api.app)
        # The same bytes into the second workspace: dedup gives one paper row.
        again = await api.client.post(
            f"/v1/workspaces/{second}/papers/upload",
            files={"file": ("a.pdf", pdf, "application/pdf")},
        )
        assert again.json()["paper_id"] == paper_id
        assert again.json()["deduplicated"] is True

        removed = await api.client.delete(f"/v1/workspaces/{first}/papers/{paper_id}")
        assert removed.json() == {
            "paper_id": paper_id,
            "removed": True,
            "purged": False,
            "vectors_deleted": 0,
        }
        # Still retrievable from the workspace that kept it.
        assert await api.app.state.index.count(paper_ids=[paper_id]) > 0
        survivors = await api.client.get(f"/v1/workspaces/{second}/papers")
        assert [p["paper_id"] for p in survivors.json()] == [paper_id]


@needs_qdrant
class TestTheChatSurface:
    async def test_asking_an_empty_workspace_is_409_not_a_broken_stream(self, api: Harness) -> None:
        # The reason phase 6 made `stream()` a coroutine: validation has to run
        # before the response commits, or this would be a 200 whose body is an
        # error (§5.5, FR-A2).
        workspace_id = (await api.client.post("/v1/workspaces", json={"name": "Empty"})).json()[
            "id"
        ]
        conversation = await api.client.post(
            f"/v1/workspaces/{workspace_id}/conversations", json={"title": None}
        )
        response = await api.client.post(
            f"/v1/workspaces/{workspace_id}/conversations/{conversation.json()['conversation_id']}/messages",
            json={"question": "anything"},
        )
        assert response.status_code == 409
        assert response.json()["type"] == "/problems/workspace-not-queryable"
        assert response.headers["content-type"].startswith("application/problem+json")

    async def test_a_whitespace_only_question_is_422_not_500(self, api: Harness) -> None:
        # The domain rejects a blank query with an `InvariantViolation`, which is
        # correctly a 500 — a violated invariant means a bug. Three spaces is not
        # a bug, so the boundary has to refuse it first (NFR-SEC2).
        workspace_id = (await api.client.post("/v1/workspaces", json={"name": "W"})).json()["id"]
        conversation = await api.client.post(
            f"/v1/workspaces/{workspace_id}/conversations", json={"title": None}
        )
        response = await api.client.post(
            f"/v1/workspaces/{workspace_id}/conversations/{conversation.json()['conversation_id']}/messages",
            json={"question": "   "},
        )
        assert response.status_code == 422

    async def test_a_conversation_in_another_workspace_is_404(self, api: Harness) -> None:
        first = (await api.client.post("/v1/workspaces", json={"name": "A"})).json()["id"]
        second = (await api.client.post("/v1/workspaces", json={"name": "B"})).json()["id"]
        conversation = await api.client.post(
            f"/v1/workspaces/{first}/conversations", json={"title": None}
        )
        cid = conversation.json()["conversation_id"]

        # Same tenant, wrong workspace — the path has to be consistent, not just
        # the ownership.
        response = await api.client.get(f"/v1/workspaces/{second}/conversations/{cid}/messages")
        assert response.status_code == 404


@needs_qdrant
class TestOperabilityBasics:
    async def test_liveness_answers(self, api: Harness) -> None:
        response = await api.client.get("/health/live")
        assert response.status_code == 200
        assert response.json() == {"status": "ok"}

    async def test_the_openapi_document_is_generated(self, api: Harness) -> None:
        # NFR-DOC2 comes free with FastAPI, but only if the routes are actually
        # registered — which is what this asserts.
        document = (await api.client.get("/openapi.json")).json()
        paths = document["paths"]
        assert "/v1/workspaces" in paths
        assert "/v1/workspaces/{workspace_id}/papers/upload" in paths
        assert "/v1/workspaces/{workspace_id}/conversations/{conversation_id}/messages" in paths

    async def test_the_ui_is_served(self, api: Harness) -> None:
        # D37's one page. Mounted at the root, and it must not shadow the API.
        page = await api.client.get("/")
        assert page.status_code == 200
        assert "Scholiast" in page.text
        assert (await api.client.get("/v1/workspaces")).status_code == 200
