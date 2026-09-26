"""NFR-T5 — the tenant-isolation negative suite. Mandatory, non-skippable.

§5.8 calls this "the evidence behind the claim", and the claim is specific: layer
2 holds *even when a route check is forgotten*. So these tests never go through
the API. They call repositories directly with the wrong tenant's scope — which is
exactly the state the system is in when a developer adds an endpoint and omits
the authorization dependency.

The fixture is the one §5.8 specifies: two tenants with identical papers. Both
kinds of identical:

  * a **public** paper both tenants ingest — the AD-8 dedup case, one row shared
  * a **private** paper with byte-identical content — the §4.4.2 carve-out, two
    rows that must never collapse into one

Every test carries a positive control. A suite where everything returns empty
passes whether isolation works or the database is empty, and only one of those
is worth shipping.
"""

from __future__ import annotations

import subprocess
import sys
import uuid
from collections.abc import AsyncIterator, Iterator
from dataclasses import dataclass
from pathlib import Path

import pytest
import pytest_asyncio
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from testcontainers.community.postgres import PostgresContainer

from scholiast.application.ports.tenancy import SWEEPER_REASON, SystemContext, TenantContext
from scholiast.domain.chunking import CHUNKER_VERSION
from scholiast.domain.paper import (
    Chunk,
    ChunkRole,
    Paper,
    PaperVisibility,
    Provenance,
    SectionKind,
)
from scholiast.domain.workspace import PaperState, Workspace, WorkspaceState
from scholiast.infrastructure.db.repositories import (
    SqlChunkRepository,
    SqlPaperRepository,
    SqlWorkspaceRepository,
)

pytestmark = [pytest.mark.integration]

ROOT = Path(__file__).resolve().parents[2]

#: The byte-identical upload. Both tenants "upload" this, and the carve-out says
#: they must end up with separate papers regardless.
SHARED_UPLOAD_HASH = "identical-manuscript-bytes"
SHARED_PUBLIC_HASH = "open-access-paper-bytes"


#: Migration 0005 made `users.password_hash` NOT NULL (D85). These rows exist
#: to own data, never to log in, so they get a string that is not a valid
#: Argon2 encoding: `Argon2idHasher.verify` returns False for it against any
#: password, so the account is unreachable by construction rather than by
#: nobody knowing the password.
UNUSABLE_HASH = "not-a-real-argon2-hash"


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


@dataclass
class Tenant:
    """One tenant and everything it owns."""

    scope: TenantContext
    workspace_id: str
    private_paper_id: str
    chunk_id: str


@dataclass
class World:
    alice: Tenant
    bob: Tenant
    #: One row, reachable by both — the dedup case.
    public_paper_id: str
    public_chunk_id: str


@pytest_asyncio.fixture
async def session(postgres: str) -> AsyncIterator[AsyncSession]:
    engine = create_async_engine(postgres)
    factory = async_sessionmaker(engine, expire_on_commit=False)
    async with factory() as s:
        yield s
        await s.rollback()
    await engine.dispose()


async def _seed_user(session: AsyncSession, email: str) -> str:
    user_id = uuid.uuid4()
    await session.execute(
        text("INSERT INTO users (id, email, password_hash) VALUES (:id, :email, :hash)"),
        {"id": user_id, "email": email, "hash": UNUSABLE_HASH},
    )
    return str(user_id)


async def _seed_section(session: AsyncSession, paper_id: str) -> str:
    section_id = uuid.uuid4()
    await session.execute(
        text(
            "INSERT INTO sections (id, paper_id, kind, raw_title, path, order_index, "
            "page_start, page_end, text) VALUES (:id, :paper, 'method', 'Method', "
            "'Method', 0, 1, 3, 'the canonical section text for this paper')"
        ),
        {"id": section_id, "paper": uuid.UUID(paper_id)},
    )
    return str(section_id)


def _chunk(paper_id: str, suffix: str) -> Chunk:
    parent = f"{paper_id}:p0"
    return Chunk(
        id=f"{parent}:c{suffix}",
        text="a claim that must not leak across tenants",
        role=ChunkRole.CHILD,
        ordinal=1,
        provenance=Provenance(
            paper_id=paper_id,
            section_path="Method",
            section_kind=SectionKind.METHOD,
            page=1,
            char_start=4,
            char_end=13,
        ),
        chunker_version=CHUNKER_VERSION,
        token_count=12,
        parent_id=parent,
    )


async def _parent_chunk(session: AsyncSession, paper_id: str, section_id: str) -> None:
    """The parent window a child must reference (ck_chunks_child_has_parent)."""
    await session.execute(
        text(
            "INSERT INTO chunks (id, paper_id, section_id, role, ordinal, text, "
            "token_count, page, char_start, char_end, chunker_version) VALUES "
            "(:id, :paper, :section, 'parent', 0, 'window', 40, 1, 0, 40, :ver)"
        ),
        {
            "id": f"{paper_id}:p0",
            "paper": uuid.UUID(paper_id),
            "section": uuid.UUID(section_id),
            "ver": CHUNKER_VERSION,
        },
    )


@pytest_asyncio.fixture
async def world(session: AsyncSession) -> World:
    """Two tenants, identical papers — the fixture §5.8 specifies."""
    alice_id = await _seed_user(session, f"alice-{uuid.uuid4()}@example.com")
    bob_id = await _seed_user(session, f"bob-{uuid.uuid4()}@example.com")
    alice_scope = TenantContext(user_id=alice_id)
    bob_scope = TenantContext(user_id=bob_id)

    tenants: dict[str, Tenant] = {}
    public_paper_id = ""
    public_chunk_id = ""

    for label, owner, scope in (("alice", alice_id, alice_scope), ("bob", bob_id, bob_scope)):
        workspaces = SqlWorkspaceRepository(session, scope)
        papers = SqlPaperRepository(session, scope)
        chunks = SqlChunkRepository(session, scope)

        workspace = await workspaces.create(
            Workspace(
                id=str(uuid.uuid4()),
                owner_id=owner,
                name=f"{label}'s workspace",
                state=WorkspaceState.ACTIVE,
            )
        )

        # The private upload — identical bytes for both tenants.
        private = await papers.add(
            Paper(
                id=str(uuid.uuid4()),
                content_hash=SHARED_UPLOAD_HASH,
                visibility=PaperVisibility.PRIVATE,
                owner_id=owner,
                title="Unpublished Manuscript",
                parser_version="grobid-crf/0.8.2.1",
            )
        )
        await papers.add_to_workspace(workspace.id, private.id, PaperState.INDEXED)
        section = await _seed_section(session, private.id)
        await _parent_chunk(session, private.id, section)
        await chunks.add_many([_chunk(private.id, label)], section_ids={"Method": section})

        # The public paper: created once, then reached by both via membership.
        if not public_paper_id:
            public = await papers.add(
                Paper(
                    id=str(uuid.uuid4()),
                    content_hash=SHARED_PUBLIC_HASH,
                    visibility=PaperVisibility.PUBLIC,
                    title="An Open Access Paper",
                    parser_version="grobid-crf/0.8.2.1",
                )
            )
            public_paper_id = public.id
            public_section = await _seed_section(session, public_paper_id)
            await _parent_chunk(session, public_paper_id, public_section)
            pub_chunk = _chunk(public_paper_id, "pub")
            await chunks.add_many([pub_chunk], section_ids={"Method": public_section})
            public_chunk_id = pub_chunk.id

        await papers.add_to_workspace(workspace.id, public_paper_id, PaperState.INDEXED)
        await session.flush()

        tenants[label] = Tenant(
            scope=scope,
            workspace_id=workspace.id,
            private_paper_id=private.id,
            chunk_id=f"{private.id}:p0:c{label}",
        )

    return World(
        alice=tenants["alice"],
        bob=tenants["bob"],
        public_paper_id=public_paper_id,
        public_chunk_id=public_chunk_id,
    )


class TestWorkspaceIsolation:
    async def test_a_non_owner_cannot_read_a_workspace(
        self, session: AsyncSession, world: World
    ) -> None:
        # The direct-object-reference attack, with layer 1 assumed absent.
        assert (
            await SqlWorkspaceRepository(session, world.alice.scope).get(world.alice.workspace_id)
            is not None
        ), "positive control: the owner must be able to read it"

        assert (
            await SqlWorkspaceRepository(session, world.bob.scope).get(world.alice.workspace_id)
        ) is None

    async def test_listing_shows_only_your_own_workspaces(
        self, session: AsyncSession, world: World
    ) -> None:
        listed = await SqlWorkspaceRepository(session, world.bob.scope).list_all()
        assert [w.id for w in listed] == [world.bob.workspace_id]

    async def test_a_non_owner_cannot_rename_a_workspace(
        self, session: AsyncSession, world: World
    ) -> None:
        # A scoped UPDATE that matches nothing must report failure rather than
        # succeeding silently — otherwise the API answers 200 for a no-op.
        assert not await SqlWorkspaceRepository(session, world.bob.scope).rename(
            world.alice.workspace_id, "stolen"
        )
        still = await SqlWorkspaceRepository(session, world.alice.scope).get(
            world.alice.workspace_id
        )
        assert still is not None
        assert still.name == "alice's workspace"

    async def test_a_non_owner_cannot_delete_a_workspace(
        self, session: AsyncSession, world: World
    ) -> None:
        assert not await SqlWorkspaceRepository(session, world.bob.scope).soft_delete(
            world.alice.workspace_id
        )
        assert (
            await SqlWorkspaceRepository(session, world.alice.scope).get(world.alice.workspace_id)
        ) is not None

    async def test_a_soft_deleted_workspace_stops_serving_reads(
        self, session: AsyncSession, world: World
    ) -> None:
        # §5.6.4 sweeps asynchronously, so rows outlive the user's action. They
        # must not keep answering queries in the interval.
        repo = SqlWorkspaceRepository(session, world.alice.scope)
        assert await repo.soft_delete(world.alice.workspace_id)
        assert await repo.get(world.alice.workspace_id) is None
        assert await repo.list_all() == []


class TestPaperIsolation:
    async def test_a_non_owner_cannot_read_a_private_paper(
        self, session: AsyncSession, world: World
    ) -> None:
        assert (
            await SqlPaperRepository(session, world.alice.scope).get(world.alice.private_paper_id)
            is not None
        ), "positive control"

        assert (
            await SqlPaperRepository(session, world.bob.scope).get(world.alice.private_paper_id)
        ) is None

    async def test_identical_uploads_stay_two_separate_papers(
        self, session: AsyncSession, world: World
    ) -> None:
        # The §4.4.2 carve-out at the repository level: byte-identical
        # manuscripts from two tenants must never collapse into one shared row.
        assert world.alice.private_paper_id != world.bob.private_paper_id

        alice_view = await SqlPaperRepository(session, world.alice.scope).find_by_content_hash(
            SHARED_UPLOAD_HASH
        )
        bob_view = await SqlPaperRepository(session, world.bob.scope).find_by_content_hash(
            SHARED_UPLOAD_HASH
        )
        assert alice_view is not None
        assert bob_view is not None
        # Each tenant's dedup lookup finds *their own* row, never the other's.
        assert alice_view.id == world.alice.private_paper_id
        assert bob_view.id == world.bob.private_paper_id

    async def test_dedup_lookup_finds_a_shared_public_paper(
        self, session: AsyncSession, world: World
    ) -> None:
        # The other half of FR-I7: re-adding an open-access paper to a different
        # workspace must not re-parse it, which requires seeing the registry
        # entry. This is the deliberate non-membership-scoped read, and it is
        # safe because only public rows are visible through it.
        found = await SqlPaperRepository(session, world.bob.scope).find_by_content_hash(
            SHARED_PUBLIC_HASH
        )
        assert found is not None
        assert found.id == world.public_paper_id

    async def test_a_public_paper_is_reachable_only_through_membership(
        self, session: AsyncSession, world: World
    ) -> None:
        # Being public is not the same as being retrievable. §4.4.2 option B put
        # isolation on "the caller's permitted paper_id set", so a tenant with no
        # membership sees nothing even for an open-access paper.
        carol_id = await _seed_user(session, f"carol-{uuid.uuid4()}@example.com")
        carol = SqlPaperRepository(session, TenantContext(user_id=carol_id))

        assert await carol.get(world.public_paper_id) is None
        assert (
            await SqlPaperRepository(session, world.bob.scope).get(world.public_paper_id)
        ) is not None, "positive control: Bob has it in a workspace"

    async def test_a_non_owner_cannot_graft_a_paper_into_another_workspace(
        self, session: AsyncSession, world: World
    ) -> None:
        # Write-side isolation. Without the ownership check on the target
        # workspace, a caller could inject a paper into someone else's corpus and
        # change the answers they get.
        assert (
            await SqlPaperRepository(session, world.bob.scope).add_to_workspace(
                world.alice.workspace_id, world.bob.private_paper_id, PaperState.INDEXED
            )
        ) is None

    async def test_a_non_owner_cannot_change_membership_state(
        self, session: AsyncSession, world: World
    ) -> None:
        assert not await SqlPaperRepository(session, world.bob.scope).set_membership_state(
            world.alice.workspace_id, world.alice.private_paper_id, PaperState.REMOVED
        )

    async def test_a_non_owner_cannot_remove_a_paper(
        self, session: AsyncSession, world: World
    ) -> None:
        assert not await SqlPaperRepository(session, world.bob.scope).remove_from_workspace(
            world.alice.workspace_id, world.alice.private_paper_id
        )


class TestRetrievalScopeIsolation:
    """FR-A4: the id set layer 3's vector filter is built from (D36)."""

    async def test_queryable_ids_are_empty_for_a_non_owner(
        self, session: AsyncSession, world: World
    ) -> None:
        # This is the single most load-bearing query in the isolation story: D36
        # builds the mandatory Qdrant filter from it. If it ever returned another
        # tenant's papers, every downstream filter would be correctly applied to
        # the wrong set.
        owner_ids = await SqlPaperRepository(session, world.alice.scope).queryable_paper_ids(
            world.alice.workspace_id
        )
        assert set(owner_ids) == {world.alice.private_paper_id, world.public_paper_id}

        intruder = await SqlPaperRepository(session, world.bob.scope).queryable_paper_ids(
            world.alice.workspace_id
        )
        assert intruder == ()

    async def test_only_indexed_papers_are_queryable(
        self, session: AsyncSession, world: World
    ) -> None:
        # FR-W5: a workspace answers over what is ready. An in-flight paper must
        # not enter the retrieval scope, or the answer cites chunks that do not
        # exist yet.
        papers = SqlPaperRepository(session, world.alice.scope)
        assert await papers.set_membership_state(
            world.alice.workspace_id, world.alice.private_paper_id, PaperState.EMBEDDING
        )
        remaining = await papers.queryable_paper_ids(world.alice.workspace_id)
        assert remaining == (world.public_paper_id,)


class TestChunkIsolation:
    async def test_a_non_owner_cannot_read_chunks_by_id(
        self, session: AsyncSession, world: World
    ) -> None:
        # FR-A2 names chunks explicitly. Guessing a chunk id is easy — they are
        # derived deterministically from the paper id — so this must hold on the
        # id path, not only on the search path.
        assert await SqlChunkRepository(session, world.alice.scope).get_many(
            [world.alice.chunk_id]
        ), "positive control"

        assert (
            await SqlChunkRepository(session, world.bob.scope).get_many([world.alice.chunk_id])
        ) == []

    async def test_a_mixed_id_batch_returns_only_the_reachable_chunks(
        self, session: AsyncSession, world: World
    ) -> None:
        # The realistic leak: a batch of ids where most are legitimate. Filtering
        # must be per-row, not "reject the batch" or "trust the batch".
        got = await SqlChunkRepository(session, world.bob.scope).get_many(
            [world.bob.chunk_id, world.alice.chunk_id, world.public_chunk_id]
        )
        assert {c.id for c in got} == {world.bob.chunk_id, world.public_chunk_id}

    async def test_a_non_owner_cannot_list_a_papers_chunks(
        self, session: AsyncSession, world: World
    ) -> None:
        assert (
            await SqlChunkRepository(session, world.bob.scope).list_for_paper(
                world.alice.private_paper_id
            )
        ) == []

    async def test_chunk_provenance_survives_the_round_trip(
        self, session: AsyncSession, world: World
    ) -> None:
        # D42: the repository must reconstruct section_kind and section_path from
        # the join, because Chunk.contextual_header() depends on section_kind to
        # mark related-work chunks. Losing it in persistence would disable the
        # misattribution defence for anything loaded from the database.
        (chunk,) = await SqlChunkRepository(session, world.alice.scope).get_many(
            [world.alice.chunk_id]
        )
        assert chunk.provenance.section_kind is SectionKind.METHOD
        assert chunk.provenance.section_path == "Method"
        assert chunk.provenance.page == 1
        assert chunk.chunker_version == CHUNKER_VERSION


class TestSystemScope:
    """The sanctioned bypass — rare, named, and greppable."""

    async def test_system_context_sees_across_tenants(
        self, session: AsyncSession, world: World
    ) -> None:
        # The reconciliation sweeper (§5.6.4) legitimately operates across
        # tenants. If this did not work, orphaned-vector cleanup could not run.
        sweeper = SystemContext(reason=SWEEPER_REASON)
        assert (
            await SqlWorkspaceRepository(session, sweeper).get(world.alice.workspace_id)
        ) is not None
        assert (
            await SqlPaperRepository(session, sweeper).get(world.alice.private_paper_id)
        ) is not None

    async def test_a_system_context_must_state_a_reason(self) -> None:
        # Turns an invisible privilege escalation into a readable one.
        with pytest.raises(ValueError, match="stated reason"):
            SystemContext(reason="  ")

    async def test_a_tenant_context_cannot_be_empty(self) -> None:
        # An empty tenant id would match nothing — or everything, depending on
        # how the adapter builds its predicate. Neither is acceptable.
        with pytest.raises(ValueError, match="non-empty user_id"):
            TenantContext(user_id="")


class TestRepositoriesCannotBeBuiltUnscoped:
    def test_constructing_without_a_scope_is_a_type_error(self) -> None:
        # §5.8's claim is "constructing a repository without tenant context is
        # impossible by type signature". mypy enforces that statically; this
        # asserts the runtime signature agrees, so the claim cannot quietly
        # become false through a default argument.
        import inspect

        for repo in (SqlWorkspaceRepository, SqlPaperRepository, SqlChunkRepository):
            params = inspect.signature(repo.__init__).parameters
            assert "scope" in params, f"{repo.__name__} takes no scope"
            assert params["scope"].default is inspect.Parameter.empty, (
                f"{repo.__name__} has a default scope — isolation became opt-in"
            )
