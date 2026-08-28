"""The schema's guarantees, asserted against a real Postgres.

These tests exist because §4.4.2's privacy carve-out is claimed to be enforced
*by the database*. That claim is only worth making if something tries to violate
it and gets rejected — so each test here attempts the violation.

Migrations are what create the schema, never `metadata.create_all()`. Testing
against `create_all` would verify the models and leave the migration — the thing
that actually runs against a real database — unexercised, which is precisely the
path that breaks in production.
"""

from __future__ import annotations

import subprocess
import sys
import uuid
from collections.abc import AsyncIterator, Iterator
from pathlib import Path

import pytest
import pytest_asyncio
from sqlalchemy import text
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncConnection, create_async_engine
from testcontainers.community.postgres import PostgresContainer

pytestmark = [pytest.mark.integration]

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def postgres() -> Iterator[str]:
    """A throwaway Postgres with the migrations applied."""
    with PostgresContainer("postgres:17-alpine", driver="asyncpg") as container:
        url = container.get_connection_url()
        result = subprocess.run(  # noqa: S603 - fixed argv, no shell
            [
                sys.executable,
                "-m",
                "alembic",
                "-x",
                f"db_url={url}",
                "upgrade",
                "head",
            ],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode != 0:
            pytest.fail(f"alembic upgrade failed:\n{result.stdout}\n{result.stderr}")
        yield url


@pytest_asyncio.fixture
async def conn(postgres: str) -> AsyncIterator[AsyncConnection]:
    engine = create_async_engine(postgres)
    async with engine.connect() as connection:
        yield connection
    await engine.dispose()


async def _user(conn: AsyncConnection, email: str) -> uuid.UUID:
    user_id = uuid.uuid4()
    await conn.execute(
        text("INSERT INTO users (id, email) VALUES (:id, :email)"),
        {"id": user_id, "email": email},
    )
    return user_id


async def _paper(
    conn: AsyncConnection,
    *,
    visibility: str,
    content_hash: str,
    owner_id: uuid.UUID | None,
) -> uuid.UUID:
    paper_id = uuid.uuid4()
    await conn.execute(
        text(
            "INSERT INTO papers (id, visibility, owner_id, content_hash, external_ids, "
            "title, authors, parse_status) VALUES (:id, :vis, :owner, :hash, '{}'::jsonb, "
            "'A Paper', '[]'::jsonb, 'indexed')"
        ),
        {"id": paper_id, "vis": visibility, "owner": owner_id, "hash": content_hash},
    )
    return paper_id


class TestPrivacyCarveOut:
    """§4.4.2, enforced by the database rather than trusted to the application."""

    async def test_the_same_public_paper_cannot_be_registered_twice(
        self, conn: AsyncConnection
    ) -> None:
        # Open-access papers deduplicate globally so they are parsed and
        # embedded once (AD-8). Without this index, a race between two
        # simultaneous ingestions produces two registry entries and doubles the
        # vector index for that paper.
        await _paper(conn, visibility="public", content_hash="hash-shared", owner_id=None)
        with pytest.raises(IntegrityError, match="uq_papers_public_hash"):
            await _paper(conn, visibility="public", content_hash="hash-shared", owner_id=None)
        await conn.rollback()

    async def test_two_tenants_uploading_identical_bytes_get_separate_papers(
        self, conn: AsyncConnection
    ) -> None:
        # THE privacy test. An unpublished manuscript uploaded by two users must
        # produce two rows, because a shared registry entry would be reachable
        # through the other tenant's membership. Global dedup on content hash —
        # the obvious optimisation — is exactly the leak.
        alice = await _user(conn, "alice@example.com")
        bob = await _user(conn, "bob@example.com")

        a = await _paper(conn, visibility="private", content_hash="same", owner_id=alice)
        b = await _paper(conn, visibility="private", content_hash="same", owner_id=bob)

        assert a != b
        rows = (
            (
                await conn.execute(
                    text("SELECT owner_id FROM papers WHERE content_hash = 'same'"),
                )
            )
            .scalars()
            .all()
        )
        assert sorted(str(r) for r in rows) == sorted([str(alice), str(bob)])
        await conn.rollback()

    async def test_one_tenant_uploading_the_same_bytes_twice_is_deduplicated(
        self, conn: AsyncConnection
    ) -> None:
        # The other half: within one owner, re-uploading is idempotent (FR-I7),
        # so a user who uploads the same PDF twice does not pay to parse it twice.
        alice = await _user(conn, "alice2@example.com")
        await _paper(conn, visibility="private", content_hash="dup", owner_id=alice)
        with pytest.raises(IntegrityError, match="uq_papers_private_owner_hash"):
            await _paper(conn, visibility="private", content_hash="dup", owner_id=alice)
        await conn.rollback()

    async def test_a_private_paper_must_have_an_owner(self, conn: AsyncConnection) -> None:
        # An ownerless private paper is unreachable by any tenant filter — it
        # would be invisible data that still consumes an index slot.
        with pytest.raises(IntegrityError, match="ck_papers_private_iff_owned"):
            await _paper(conn, visibility="private", content_hash="orphan", owner_id=None)
        await conn.rollback()

    async def test_a_public_paper_must_not_be_owner_scoped(self, conn: AsyncConnection) -> None:
        # The reverse direction, which a naive NOT NULL check would miss: a
        # public paper carrying an owner_id claims global sharing and private
        # ownership at once.
        alice = await _user(conn, "alice3@example.com")
        with pytest.raises(IntegrityError, match="ck_papers_private_iff_owned"):
            await _paper(conn, visibility="public", content_hash="confused", owner_id=alice)
        await conn.rollback()


class TestTaxonomiesMatchTheDomain:
    """CHECK constraints are generated from the domain enums, so they cannot drift."""

    async def test_an_unknown_section_kind_is_rejected(self, conn: AsyncConnection) -> None:
        # Prevents a parser writing a section kind nothing downstream can
        # interpret — attribution and section priors both switch on this value,
        # so an unrecognised one degrades retrieval silently.
        paper = await _paper(conn, visibility="public", content_hash="kinds", owner_id=None)
        with pytest.raises(IntegrityError, match="ck_sections_kind"):
            await conn.execute(
                text(
                    "INSERT INTO sections (id, paper_id, kind, raw_title, path, order_index, "
                    "page_start, page_end, text) VALUES (:id, :paper, 'methodology', 'M', "
                    "'M', 0, 1, 2, 'body')"
                ),
                {"id": uuid.uuid4(), "paper": paper},
            )
        await conn.rollback()

    async def test_an_unknown_paper_state_is_rejected(self, conn: AsyncConnection) -> None:
        user = await _user(conn, "states@example.com")
        workspace = uuid.uuid4()
        await conn.execute(
            text(
                "INSERT INTO workspaces (id, owner_id, name, state) "
                "VALUES (:id, :owner, 'W', 'active')"
            ),
            {"id": workspace, "owner": user},
        )
        paper = await _paper(conn, visibility="public", content_hash="states", owner_id=None)
        with pytest.raises(IntegrityError, match="ck_workspace_papers_state"):
            await conn.execute(
                text(
                    "INSERT INTO workspace_papers (workspace_id, paper_id, state) "
                    "VALUES (:ws, :paper, 'processing')"
                ),
                {"ws": workspace, "paper": paper},
            )
        await conn.rollback()


class TestChunkInvariantsSurviveIntoTheDatabase:
    """The domain's chunk rules are re-asserted in SQL, not merely in Python."""

    async def _section(self, conn: AsyncConnection) -> tuple[uuid.UUID, uuid.UUID]:
        paper = await _paper(
            conn, visibility="public", content_hash=f"c-{uuid.uuid4()}", owner_id=None
        )
        section = uuid.uuid4()
        await conn.execute(
            text(
                "INSERT INTO sections (id, paper_id, kind, raw_title, path, order_index, "
                "page_start, page_end, text) VALUES (:id, :paper, 'method', 'Method', "
                "'Method', 0, 1, 3, 'the canonical section text')"
            ),
            {"id": section, "paper": paper},
        )
        return paper, section

    async def _insert_chunk(
        self, conn: AsyncConnection, paper: uuid.UUID, section: uuid.UUID, **overrides: object
    ) -> None:
        values: dict[str, object] = {
            "id": f"chunk-{uuid.uuid4()}",
            "paper": paper,
            "section": section,
            "parent": None,
            "role": "parent",
            "ordinal": 0,
            "text": "body",
            "tokens": 10,
            "page": 1,
            "start": 0,
            "end": 5,
        }
        values.update(overrides)
        await conn.execute(
            text(
                "INSERT INTO chunks (id, paper_id, section_id, parent_id, role, ordinal, "
                "text, token_count, page, char_start, char_end, chunker_version) VALUES "
                "(:id, :paper, :section, :parent, :role, :ordinal, :text, :tokens, :page, "
                ":start, :end, 'structure-aware/1.1.0')"
            ),
            values,
        )

    async def test_a_child_chunk_without_a_parent_is_rejected(self, conn: AsyncConnection) -> None:
        # A child with no parent window cannot be parent-expanded, which is the
        # entire point of small-to-big retrieval — it would retrieve precisely
        # and then answer from a 300-token fragment.
        paper, section = await self._section(conn)
        with pytest.raises(IntegrityError, match="ck_chunks_child_has_parent"):
            await self._insert_chunk(conn, paper, section, role="child", parent=None)
        await conn.rollback()

    async def test_an_empty_character_span_is_rejected(self, conn: AsyncConnection) -> None:
        # A zero-width span resolves to no source text, so the citation it backs
        # would highlight nothing (FR-I5).
        paper, section = await self._section(conn)
        with pytest.raises(IntegrityError, match="ck_chunks_span_non_empty"):
            await self._insert_chunk(conn, paper, section, start=40, end=40)
        await conn.rollback()

    async def test_a_zero_page_is_rejected(self, conn: AsyncConnection) -> None:
        # Pages are 1-indexed; a 0 means a parser forgot to convert and every
        # citation from that paper points one page early.
        paper, section = await self._section(conn)
        with pytest.raises(IntegrityError, match="ck_chunks_page_one_indexed"):
            await self._insert_chunk(conn, paper, section, page=0)
        await conn.rollback()

    async def test_offsets_resolve_against_the_section_text(self, conn: AsyncConnection) -> None:
        # D42's round trip: a chunk's span indexes `sections.text`, so resolving
        # a citation is this one join plus a slice. This is the property the
        # whole provenance design exists to provide.
        paper, section = await self._section(conn)
        await self._insert_chunk(conn, paper, section, start=4, end=13, text="canonical")
        resolved = (
            await conn.execute(
                text(
                    "SELECT substring(s.text FROM c.char_start + 1 FOR c.char_end - c.char_start) "
                    "FROM chunks c JOIN sections s ON s.id = c.section_id "
                    "WHERE c.section_id = :section"
                ),
                {"section": section},
            )
        ).scalar_one()
        assert resolved == "canonical"
        await conn.rollback()


class TestFailureTaxonomy:
    async def test_a_failed_paper_must_carry_a_failure_class(self, conn: AsyncConnection) -> None:
        # A failure without a class cannot produce a retry decision or a user
        # message, which defeats the taxonomy in doc 4 §4.4.1 entirely.
        user = await _user(conn, "fail@example.com")
        workspace = uuid.uuid4()
        await conn.execute(
            text(
                "INSERT INTO workspaces (id, owner_id, name, state) "
                "VALUES (:id, :owner, 'W', 'active')"
            ),
            {"id": workspace, "owner": user},
        )
        paper = await _paper(conn, visibility="public", content_hash="failing", owner_id=None)
        with pytest.raises(IntegrityError, match="ck_workspace_papers_failure_classified"):
            await conn.execute(
                text(
                    "INSERT INTO workspace_papers (workspace_id, paper_id, state) "
                    "VALUES (:ws, :paper, 'failed')"
                ),
                {"ws": workspace, "paper": paper},
            )
        await conn.rollback()
