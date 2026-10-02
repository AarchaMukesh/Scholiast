"""FR-W3's acceptance test: zero residue in Postgres, Qdrant and object storage.

> "Deleting a workspace leaves zero residue in Postgres, Qdrant, and object
> storage." — doc 8, M3 acceptance criteria

Real Postgres, real Qdrant, a real directory. The state is **built directly
rather than ingested**, because what is under test is the cascade and not the
pipeline: GROBID would add three seconds and a service dependency to every case
without changing anything the cascade sees. `test_ingestion_pipeline` already
proves the rows this file deletes are the rows ingestion writes.

**Residue is asserted by counting rows in each store, never by trusting a return
value.** A cascade that reported `papers_purged=1` while leaving the vectors is
exactly the failure mode FR-W3 exists to catch, and it is indistinguishable from
success if the assertion reads the result object.

The two-tenant setup is not decoration. A shared paper is the only thing that
distinguishes "delete everything reachable" from "delete what nothing else
holds", and getting that wrong breaks a tenant who deleted nothing (AD-8).
"""

from __future__ import annotations

import subprocess
import sys
import uuid
from collections.abc import AsyncIterator, Iterator
from pathlib import Path

import httpx
import pytest
import pytest_asyncio
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from testcontainers.community.postgres import PostgresContainer

from scholiast.application.ports.storage import paper_prefix, source_key, tei_key
from scholiast.application.ports.tenancy import (
    SWEEPER_REASON,
    SystemContext,
    TenantContext,
    TenantIsolationViolation,
)
from scholiast.application.ports.vector import VectorRecord
from scholiast.domain.paper import SectionKind
from scholiast.infrastructure.db.repositories import SqlPaperRepository, SqlWorkspaceRepository
from scholiast.infrastructure.storage.local import LocalObjectStore
from scholiast.infrastructure.vector.qdrant import QdrantVectorIndex
from scholiast.workers.maintenance import MaintenanceWorker

pytestmark = [pytest.mark.integration]

ROOT = Path(__file__).resolve().parents[2]
QDRANT_URL = "http://localhost:6333"
DIMENSIONS = 32
PARSER_VERSION = "grobid-crf-0.8.2.1/tei/1.1.0"
#: A second parser generation, kept on purpose by D49 so a re-chunk ablation can
#: read the bytes that produced today's chunks. It is also the blob a purge that
#: reconstructs one `tei_key` silently leaves behind.
OLDER_PARSER_VERSION = "grobid-crf-0.8.1.0/tei/1.0.0"

#: Migration 0005 made `users.password_hash` NOT NULL (D85). These rows own data
#: and never log in, so they get a string that is not a valid Argon2 encoding.
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


@pytest_asyncio.fixture
async def sessions(postgres: str) -> AsyncIterator[async_sessionmaker[AsyncSession]]:
    engine = create_async_engine(postgres)
    factory = async_sessionmaker(engine, expire_on_commit=False)
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
            "users",
        ):
            await connection.execute(text(f"DELETE FROM {table}"))  # noqa: S608 - fixed list
    yield factory
    await engine.dispose()


@pytest_asyncio.fixture
async def index() -> AsyncIterator[QdrantVectorIndex]:
    built = QdrantVectorIndex(
        QDRANT_URL, collection=f"cascade_{uuid.uuid4().hex[:8]}", dimensions=DIMENSIONS
    )
    await built.ensure_ready()
    yield built
    await built._client.delete_collection(built.collection)
    await built.close()


class World:
    """Two tenants, three papers, and every store populated for real."""

    def __init__(
        self,
        sessions: async_sessionmaker[AsyncSession],
        index: QdrantVectorIndex,
        objects: Path,
    ) -> None:
        self.sessions = sessions
        self.index = index
        self.store = LocalObjectStore(objects)
        self.worker = MaintenanceWorker(sessions=sessions, index=index, store=self.store)
        self.alice = str(uuid.uuid4())
        self.bob = str(uuid.uuid4())
        self.alice_workspace = str(uuid.uuid4())
        self.bob_workspace = str(uuid.uuid4())
        #: Only Alice holds it.
        self.private_paper = str(uuid.uuid4())
        #: Alice AND Bob hold it — AD-8's shared registry.
        self.shared_paper = str(uuid.uuid4())
        #: Only Bob holds it; a control that must survive untouched.
        self.bob_paper = str(uuid.uuid4())

    async def seed(self) -> None:
        async with self.sessions() as session:
            for user, email in ((self.alice, "alice"), (self.bob, "bob")):
                await session.execute(
                    text("INSERT INTO users (id, email, password_hash) VALUES (:i, :e, :h)"),
                    {"i": user, "e": f"{email}-{user[:8]}@test.local", "h": UNUSABLE_HASH},
                )
            for workspace, owner, name in (
                (self.alice_workspace, self.alice, "Alice"),
                (self.bob_workspace, self.bob, "Bob"),
            ):
                await session.execute(
                    text(
                        "INSERT INTO workspaces (id, owner_id, name, state) "
                        "VALUES (:i, :o, :n, 'active')"
                    ),
                    {"i": workspace, "o": owner, "n": name},
                )
            await session.commit()

        await self._paper(self.private_paper, owner=self.alice, held_by=[self.alice_workspace])
        await self._paper(
            self.shared_paper,
            owner=None,
            held_by=[self.alice_workspace, self.bob_workspace],
        )
        await self._paper(self.bob_paper, owner=self.bob, held_by=[self.bob_workspace])

        # A conversation in Alice's workspace: FR-W3 says every store, and
        # conversations are the rows that only the FK cascade removes.
        async with self.sessions() as session:
            await session.execute(
                text("INSERT INTO conversations (id, workspace_id, title) VALUES (:i, :w, 'chat')"),
                {"i": str(uuid.uuid4()), "w": self.alice_workspace},
            )
            await session.commit()

    async def _paper(self, paper_id: str, *, owner: str | None, held_by: list[str]) -> None:
        """One paper as ingestion leaves it: row, section, chunk, vector, blobs."""
        section_id = str(uuid.uuid4())
        async with self.sessions() as session:
            await session.execute(
                text(
                    "INSERT INTO papers (id, visibility, owner_id, content_hash, title, "
                    "parse_status, parser_version, tei_key) VALUES (:i, :v, :o, :h, :t, "
                    "'parsed', :pv, :tk)"
                ),
                {
                    "i": paper_id,
                    "v": "private" if owner else "public",
                    "o": owner,
                    "h": paper_id.replace("-", "")[:32] + "0" * 32,
                    "t": f"Paper {paper_id[:8]}",
                    "pv": PARSER_VERSION,
                    "tk": tei_key(paper_id, PARSER_VERSION),
                },
            )
            await session.execute(
                text(
                    "INSERT INTO sections (id, paper_id, kind, raw_title, path, order_index, "
                    "page_start, page_end, text) VALUES (:i, :p, 'method', 'M', 'M', 0, 1, 1, 'x')"
                ),
                {"i": section_id, "p": paper_id},
            )
            await session.execute(
                text(
                    "INSERT INTO chunks (id, paper_id, section_id, role, ordinal, text, "
                    "token_count, page, char_start, char_end, chunker_version) "
                    "VALUES (:i, :p, :s, 'parent', 0, 'body', 2, 1, 0, 1, 'test/1.0.0')"
                ),
                {"i": f"{paper_id}:0", "p": paper_id, "s": section_id},
            )
            for workspace in held_by:
                await session.execute(
                    text(
                        "INSERT INTO workspace_papers (workspace_id, paper_id, state) "
                        "VALUES (:w, :p, 'indexed')"
                    ),
                    {"w": workspace, "p": paper_id},
                )
            await session.commit()

        await self.index.upsert(
            [
                VectorRecord(
                    chunk_id=f"{paper_id}:0",
                    paper_id=paper_id,
                    vector=tuple(0.1 for _ in range(DIMENSIONS)),
                    section_kind=SectionKind.METHOD,
                    chunker_version="test/1.0.0",
                    embedder_version="test/1.0.0",
                )
            ]
        )
        await self.store.put(source_key(paper_id), b"%PDF-1.4", content_type="application/pdf")
        await self.store.put(tei_key(paper_id, PARSER_VERSION), b"<TEI/>", content_type="text/xml")
        # The second generation D49 keeps. Nothing in the paper row names it.
        await self.store.put(
            tei_key(paper_id, OLDER_PARSER_VERSION), b"<TEI/>", content_type="text/xml"
        )

    # -- what the stores actually contain ----------------------------------

    async def mark_deleted(self, workspace_id: str, owner: str) -> bool:
        async with self.sessions() as session:
            repo = SqlWorkspaceRepository(session, TenantContext(user_id=owner))
            marked = await repo.soft_delete(workspace_id)
            await session.commit()
        return marked

    async def scalar(self, sql: str, **params: object) -> int:
        async with self.sessions() as session:
            return int((await session.execute(text(sql), params)).scalar_one())

    async def paper_rows(self, paper_id: str) -> int:
        return await self.scalar(
            "SELECT count(*) FROM papers WHERE id = CAST(:p AS uuid)", p=paper_id
        )

    async def chunk_rows(self, paper_id: str) -> int:
        return await self.scalar(
            "SELECT count(*) FROM chunks WHERE paper_id = CAST(:p AS uuid)", p=paper_id
        )

    async def section_rows(self, paper_id: str) -> int:
        return await self.scalar(
            "SELECT count(*) FROM sections WHERE paper_id = CAST(:p AS uuid)", p=paper_id
        )

    async def blobs(self, paper_id: str) -> list[str]:
        return await self.store.list_keys(paper_prefix(paper_id))


@pytest_asyncio.fixture
async def world(
    sessions: async_sessionmaker[AsyncSession], index: QdrantVectorIndex, tmp_path: Path
) -> AsyncIterator[World]:
    built = World(sessions, index, tmp_path / "objects")
    await built.seed()
    yield built


@needs_qdrant
class TestDeletingAWorkspaceLeavesZeroResidue:
    """M3's acceptance criterion, asserted store by store."""

    async def test_postgres_keeps_nothing_of_the_unshared_paper(self, world: World) -> None:
        await world.mark_deleted(world.alice_workspace, world.alice)

        await world.worker.delete_workspace(world.alice_workspace)

        assert await world.paper_rows(world.private_paper) == 0
        # Sections and chunks go by FK cascade, not by the use case naming
        # them. Asserting them separately is what proves the cascade fired
        # rather than that someone remembered to delete two more tables.
        assert await world.section_rows(world.private_paper) == 0
        assert await world.chunk_rows(world.private_paper) == 0

    async def test_qdrant_keeps_nothing_of_the_unshared_paper(self, world: World) -> None:
        await world.mark_deleted(world.alice_workspace, world.alice)

        await world.worker.delete_workspace(world.alice_workspace)

        assert await world.index.count(paper_ids=[world.private_paper]) == 0

    async def test_object_storage_keeps_nothing_of_the_unshared_paper(self, world: World) -> None:
        # Including the TEI of the OLDER parser generation, which no row names.
        # A purge built from `source_key` plus the recorded `tei_key` passes
        # every other assertion in this class and leaks this one file.
        await world.mark_deleted(world.alice_workspace, world.alice)

        await world.worker.delete_workspace(world.alice_workspace)

        assert await world.blobs(world.private_paper) == []

    async def test_the_workspace_and_its_conversations_are_gone(self, world: World) -> None:
        await world.mark_deleted(world.alice_workspace, world.alice)

        await world.worker.delete_workspace(world.alice_workspace)

        assert (
            await world.scalar(
                "SELECT count(*) FROM workspaces WHERE id = CAST(:w AS uuid)",
                w=world.alice_workspace,
            )
            == 0
        )
        assert (
            await world.scalar(
                "SELECT count(*) FROM conversations WHERE workspace_id = CAST(:w AS uuid)",
                w=world.alice_workspace,
            )
            == 0
        )

    async def test_the_whole_prefix_is_empty_afterwards(self, world: World) -> None:
        # The residue question asked the way FR-W3 asks it: not "are the keys we
        # thought of gone" but "is anything left under this paper at all".
        await world.mark_deleted(world.alice_workspace, world.alice)

        await world.worker.delete_workspace(world.alice_workspace)

        leftovers = [
            key
            for key in await world.store.list_keys("papers/")
            if key.startswith(paper_prefix(world.private_paper))
        ]
        assert leftovers == []


@needs_qdrant
class TestTheOtherTenantIsUntouched:
    """AD-8's cost, paid correctly: sharing must not make deletion contagious."""

    async def test_the_shared_paper_survives_in_every_store(self, world: World) -> None:
        await world.mark_deleted(world.alice_workspace, world.alice)

        await world.worker.delete_workspace(world.alice_workspace)

        assert await world.paper_rows(world.shared_paper) == 1
        assert await world.chunk_rows(world.shared_paper) == 1
        assert await world.index.count(paper_ids=[world.shared_paper]) == 1
        assert len(await world.blobs(world.shared_paper)) == 3

    async def test_bobs_membership_and_his_own_paper_survive(self, world: World) -> None:
        await world.mark_deleted(world.alice_workspace, world.alice)

        await world.worker.delete_workspace(world.alice_workspace)

        assert (
            await world.scalar(
                "SELECT count(*) FROM workspace_papers WHERE workspace_id = CAST(:w AS uuid)",
                w=world.bob_workspace,
            )
            == 2
        )
        assert await world.index.count(paper_ids=[world.bob_paper]) == 1

    async def test_deleting_bobs_workspace_afterwards_finally_purges_the_shared_paper(
        self, world: World
    ) -> None:
        # The other half of AD-8: the shared paper is not immortal, it is held.
        # Once the last holder goes, so does it.
        await world.mark_deleted(world.alice_workspace, world.alice)
        await world.worker.delete_workspace(world.alice_workspace)

        await world.mark_deleted(world.bob_workspace, world.bob)
        await world.worker.delete_workspace(world.bob_workspace)

        assert await world.paper_rows(world.shared_paper) == 0
        assert await world.index.count(paper_ids=[world.shared_paper]) == 0
        assert await world.blobs(world.shared_paper) == []


@needs_qdrant
class TestAnEmptyLiveSetMeansEverythingIsAnOrphan:
    """The orphan filter's dangerous branch, against real Qdrant.

    `_orphan_filter` substitutes a sentinel when the live set is empty, so that
    `must_not MatchAny([sentinel])` excludes nothing and therefore matches the
    whole collection. That is one line of adapter cleverness standing in for a
    branch, and **a fake index cannot test it** — the fake's behaviour is the
    test's own code, so it would agree with any filter at all. Only Qdrant can
    say whether the filter it is handed means what the docstring claims.

    The failure this guards against is the natural implementation: an empty-list
    guard returning 0. That reads as "nothing to do" and is exactly backwards —
    no live papers means every vector in the index is residue.
    """

    async def test_the_count_sees_every_vector_when_postgres_has_no_papers(
        self, world: World
    ) -> None:
        # Asserted separately from the delete because the two share the filter:
        # a filter matching nothing would make BOTH report zero, and the sweep
        # would no-op while claiming success.
        assert await world.index.count_orphans([]) == 3

    async def test_a_full_live_set_finds_no_orphans(self, world: World) -> None:
        # The positive control. Without it the test above passes against a
        # filter that matches the whole collection unconditionally.
        live = [world.private_paper, world.shared_paper, world.bob_paper]

        assert await world.index.count_orphans(live) == 0
        assert await world.index.delete_orphans(live) == 0
        assert await world.index.count() == 3

    async def test_deleting_every_paper_row_removes_every_vector(self, world: World) -> None:
        async with world.sessions() as session:
            await session.execute(text("DELETE FROM papers"))
            await session.commit()

        result = await world.worker.reconcile()

        assert result.vectors_deleted == 3
        assert await world.index.count() == 0


@needs_qdrant
class TestTheCascadeIsRestartable:
    """§5.6.4 step 3: idempotent and resumable, against real stores."""

    async def test_running_it_twice_is_safe(self, world: World) -> None:
        await world.mark_deleted(world.alice_workspace, world.alice)

        first = await world.worker.delete_workspace(world.alice_workspace)
        second = await world.worker.delete_workspace(world.alice_workspace)

        assert first.swept is True
        # Refused the second time because the mark is gone — which is what stops
        # a redelivered Celery message from running a second cascade against
        # whatever now owns those ids.
        assert second.swept is False
        assert await world.paper_rows(world.shared_paper) == 1

    async def test_a_cascade_interrupted_after_the_vectors_went_still_finishes(
        self, world: World
    ) -> None:
        # Simulate the crash window D62's ordering is chosen for: vectors gone,
        # Postgres untouched. The resumed cascade must converge, not fail on
        # rows that are already consistent with a half-finished attempt.
        await world.mark_deleted(world.alice_workspace, world.alice)
        await world.index.delete_by_papers([world.private_paper])

        result = await world.worker.delete_workspace(world.alice_workspace)

        assert result.swept is True
        assert await world.paper_rows(world.private_paper) == 0
        assert await world.blobs(world.private_paper) == []


@needs_qdrant
class TestTheSweepFindsWhatTheCascadeNeverStarted:
    """D99: the mark is the work item, so a lost dispatch loses nothing."""

    async def test_a_marked_workspace_nobody_dispatched_is_swept(self, world: World) -> None:
        # The broker was down when the 202 was returned. Nothing is holding a
        # reference to this work except the row itself.
        await world.mark_deleted(world.alice_workspace, world.alice)

        result = await world.worker.reconcile()

        assert result.cascades_resumed == 1
        assert await world.paper_rows(world.private_paper) == 0
        assert await world.index.count(paper_ids=[world.private_paper]) == 0

    async def test_an_orphaned_paper_is_found_and_purged(self, world: World) -> None:
        # The D99 race made concrete: the membership went, the purge did not.
        # This paper is now reachable by no scoped query at all, so only the
        # sweep can ever see it.
        async with world.sessions() as session:
            await session.execute(
                text("DELETE FROM workspace_papers WHERE paper_id = CAST(:p AS uuid)"),
                {"p": world.private_paper},
            )
            await session.commit()

        result = await world.worker.reconcile()

        assert result.papers_purged == 1
        assert await world.paper_rows(world.private_paper) == 0
        assert await world.index.count(paper_ids=[world.private_paper]) == 0
        assert await world.blobs(world.private_paper) == []

    async def test_a_vector_whose_paper_row_vanished_is_removed(self, world: World) -> None:
        # The orphan §5.6.4 names, and the one NFR-R6 rests on: Postgres is the
        # source of truth, so a point with no row behind it does not belong.
        async with world.sessions() as session:
            await session.execute(
                text("DELETE FROM papers WHERE id = CAST(:p AS uuid)"), {"p": world.private_paper}
            )
            await session.commit()

        assert await world.index.count(paper_ids=[world.private_paper]) == 1

        result = await world.worker.reconcile()

        assert result.vectors_deleted >= 1
        assert await world.index.count(paper_ids=[world.private_paper]) == 0

    async def test_a_blob_whose_paper_row_vanished_is_removed(self, world: World) -> None:
        async with world.sessions() as session:
            await session.execute(
                text("DELETE FROM papers WHERE id = CAST(:p AS uuid)"), {"p": world.private_paper}
            )
            await session.commit()

        await world.worker.reconcile()

        assert await world.blobs(world.private_paper) == []

    async def test_a_clean_system_is_reported_as_clean(self, world: World) -> None:
        # The number has to mean something. If a healthy system reported
        # residue every pass, a real leak would be invisible in the noise.
        result = await world.worker.reconcile()

        assert result.cascades_resumed == 0
        assert result.papers_purged == 0
        assert result.vectors_deleted == 0
        assert result.blobs_deleted == 0

    async def test_the_sweep_leaves_healthy_data_alone(self, world: World) -> None:
        await world.worker.reconcile()

        for paper in (world.private_paper, world.shared_paper, world.bob_paper):
            assert await world.paper_rows(paper) == 1
            assert await world.index.count(paper_ids=[paper]) == 1
            assert len(await world.blobs(paper)) == 3


@needs_qdrant
class TestTheCascadeCannotBeAimedAtLiveData:
    """D99's guard, against real stores rather than a fake.

    The task argument arrives over a broker and the cascade runs unscoped, so
    this is the only thing standing between a stray message and a live
    workspace.
    """

    async def test_an_unmarked_workspace_is_refused(self, world: World) -> None:
        result = await world.worker.delete_workspace(world.bob_workspace)

        assert result.swept is False
        assert await world.paper_rows(world.bob_paper) == 1
        assert await world.index.count(paper_ids=[world.bob_paper]) == 1
        assert (
            await world.scalar(
                "SELECT count(*) FROM workspaces WHERE id = CAST(:w AS uuid)",
                w=world.bob_workspace,
            )
            == 1
        )

    async def test_a_garbage_id_is_refused_rather_than_raising(self, world: World) -> None:
        result = await world.worker.delete_workspace("not-a-uuid")

        assert result.swept is False


class TestTheSweepReadsAreSystemScopedOnly:
    """The cross-tenant reads carry no tenant predicate, so the scope is the check.

    Each of these would enumerate the whole system if a tenant could call it —
    `all_paper_ids` most obviously, but `list_deleting` leaks every tenant's
    pending deletions and `orphaned_paper_ids` leaks paper ids by existence.
    """

    async def test_a_tenant_cannot_list_deleting_workspaces(
        self, sessions: async_sessionmaker[AsyncSession]
    ) -> None:
        async with sessions() as session:
            repo = SqlWorkspaceRepository(session, TenantContext(user_id=str(uuid.uuid4())))
            with pytest.raises(TenantIsolationViolation):
                await repo.list_deleting()

    async def test_a_tenant_cannot_enumerate_orphaned_papers(
        self, sessions: async_sessionmaker[AsyncSession]
    ) -> None:
        async with sessions() as session:
            repo = SqlPaperRepository(session, TenantContext(user_id=str(uuid.uuid4())))
            with pytest.raises(TenantIsolationViolation):
                await repo.orphaned_paper_ids()

    async def test_a_tenant_cannot_enumerate_every_paper(
        self, sessions: async_sessionmaker[AsyncSession]
    ) -> None:
        async with sessions() as session:
            repo = SqlPaperRepository(session, TenantContext(user_id=str(uuid.uuid4())))
            with pytest.raises(TenantIsolationViolation):
                await repo.all_paper_ids()

    async def test_the_sweeper_itself_may(self, sessions: async_sessionmaker[AsyncSession]) -> None:
        # The positive control. Without it the three tests above would pass
        # against a method that raises for everyone.
        async with sessions() as session:
            scope = SystemContext(reason=SWEEPER_REASON)
            assert await SqlWorkspaceRepository(session, scope).list_deleting() == ()
            assert await SqlPaperRepository(session, scope).all_paper_ids() == ()
